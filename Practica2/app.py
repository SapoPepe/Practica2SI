import sqlite3
import json
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os
from flask import Flask, render_template, jsonify, send_file
from fpdf import FPDF
app = Flask(__name__)

with open('data_clasified.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
tickets = data['tickets_emitidos']

def leerJSON(cur, con):
    ficheroJSON = open('datos.json', 'r')
    datos = json.load(ficheroJSON)

    for elemento in datos["clientes"]:
        cur.execute("INSERT OR IGNORE INTO clientes (id_cli, nombre, provincia, telefono)"
                    "VALUES (?, ?, ?, ?)", (int(elemento['id_cli']), elemento['nombre'], elemento['provincia'], elemento['telefono']))
        con.commit()

    for elemento in datos["tickets_emitidos"]:
        #Primero insertamos todos los datos de cada ticket
        cur.execute("INSERT OR IGNORE INTO tickets_emitidos (cliente, fecha_apertura, fecha_cierre, es_mantenimiento, satisfaccion_cliente, tipo_incidencia)"
                    "VALUES (?, ?, ?, ?, ?, ?)", (int(elemento['cliente']), elemento['fecha_apertura'], elemento['fecha_cierre'], int(elemento['es_mantenimiento']), int(elemento['satisfaccion_cliente']), int(elemento['tipo_incidencia'])))

        #Segundo, insertamos todos los contactos con empleados que tenga ese ticket
        ticket_id = cur.lastrowid   #Obtener el ID del ticket
        for contacto in elemento['contactos_con_empleados']:
            cur.execute("INSERT OR IGNORE INTO contactos_con_empleados (id_ticket, id_emp, fecha, tiempo) "
                        "VALUES (?, ?, ?, ?)", (ticket_id, int(contacto['id_emp']), contacto['fecha'], float(contacto['tiempo'])))
        con.commit()

    for elemento in datos["empleados"]:
        cur.execute("INSERT OR IGNORE INTO empleados (fecha_contrato, id_emp, nivel, nombre)"
                    "VALUES (?, ?, ?, ?)", (elemento['fecha_contrato'], int(elemento['id_emp']), int(elemento['nivel']), elemento['nombre']))
        con.commit()

    for elemento in datos["tipos_incidentes"]:
        cur.execute("INSERT OR IGNORE INTO tipos_incidentes (id_incidencia, nombre)"
                    "VALUES (?, ?)", (int(elemento['id_incidencia']), elemento['nombre']))

        con.commit()

def crearBBDD():
    con = sqlite3.connect('database.db')
    cur = con.cursor()

    cur.execute("DROP TABLE IF EXISTS clientes;")
    cur.execute("DROP TABLE IF EXISTS empleados;")
    cur.execute("DROP TABLE IF EXISTS tipos_incidentes;")
    cur.execute("DROP TABLE IF EXISTS tickets_emitidos;")
    cur.execute("DROP TABLE IF EXISTS contactos_con_empleados;")

    cur.execute("CREATE TABLE IF NOT EXISTS clientes ("
                "id_cli INTEGER PRIMARY KEY,"
                "nombre TEXT,"
                "provincia TEXT,"
                "telefono TEXT"
                ");")

    cur.execute("CREATE TABLE IF NOT EXISTS empleados ("
                "id_emp INTEGER PRIMARY KEY,"
                "fecha_contrato DATE," 
                "nivel INTEGER,"
                "nombre TEXT"
                ");")

    cur.execute("CREATE TABLE IF NOT EXISTS tipos_incidentes ("
                "id_incidencia INTEGER PRIMARY KEY,"
                "nombre TEXT "
                ");")


    cur.execute("CREATE TABLE IF NOT EXISTS tickets_emitidos ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "cliente INTEGER, "
                "fecha_apertura DATE, "
                "fecha_cierre DATE, "
                "es_mantenimiento INTEGER, "
                "satisfaccion_cliente INTEGER, "
                "tipo_incidencia INTEGER, "
                "FOREIGN KEY (cliente) REFERENCES clientes(id_cli), "
                "FOREIGN KEY (tipo_incidencia) REFERENCES tipos_incidentes(id_incidencia));")


    cur.execute("CREATE TABLE IF NOT EXISTS contactos_con_empleados ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "id_ticket INTEGER, "
                "id_emp INTEGER, "
                "fecha DATE, "
                "tiempo REAL, "
                "FOREIGN KEY (id_ticket) REFERENCES tickets_emitidos(id), "
                "FOREIGN KEY (id_emp) REFERENCES empleados(id_emp)"
                ");")

    con.commit()

    leerJSON(cur, con)
    return con

def top_clientes_incidencias(x):
    clientes = {}
    for ticket in tickets:
        cliente = ticket['cliente']
        clientes[cliente] = clientes.get(cliente, 0) + 1
    sorted_clientes = sorted(clientes.items(), key=lambda item: (-item[1], item[0]))
    return sorted_clientes[:x]

def top_tipos_tiempo_resolucion(x):
    tipos = {}
    for ticket in tickets:
        tipo = ticket['tipo_incidencia']
        tiempo_total = sum(contacto['tiempo'] for contacto in ticket['contactos_con_empleados'])
        tipos[tipo] = tipos.get(tipo, 0) + tiempo_total
    sorted_tipos = sorted(tipos.items(), key=lambda item: (-item[1], item[0]))
    return sorted_tipos[:x]

@app.route('/')
def home():

    return render_template("index.html")

def calculateTopClientes(x):
    con = crearBBDD()
    dataframe = pd.read_sql("SELECT * FROM tickets_emitidos t JOIN clientes cli ON t.cliente=cli.id_cli", con)
    cliMaxInci = dataframe.groupby('id_cli').agg(
        nombre_cliente=('nombre', 'first'),
        incidencias=('id_cli', 'count')
    )

    # Ordenar y escoger los X clientes requeridos
    cliMaxIncidents_ordenado = cliMaxInci.sort_values(by='incidencias', ascending=False)[:x]

    return cliMaxIncidents_ordenado
@app.route('/top_clientes/<int:x>')
def get_top_clientes(x):
    #Ordenar y escoger los X clientes requeridos
    cliMaxIncidents_ordenado = calculateTopClientes(x)

    tabla_html = cliMaxIncidents_ordenado.to_html(classes='data')
    return render_template('top_clientes.html', tabla_html=tabla_html)

@app.route('/top_clientes/<int:x>/downloadPDF')
def generateTopClientesPDF(x):
    cliMaxIncidents_ordenado = calculateTopClientes(x)

    pdf = generatePDF()

    pdf.cell(200, 10, txt=f'Informe Top {x} Clientes con Incidencias', ln=True, align='C')
    pdf.ln(10)

    pdf.set_fill_color(200, 200, 200)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, "Nombre Cliente", border=1, align='C', fill=True)
    pdf.cell(50, 10, "Incidencias", border=1, align='C', fill=True)
    pdf.ln()

    for index, row in cliMaxIncidents_ordenado.iterrows():
        pdf.cell(100, 10, row['nombre_incidencia'], border=1, align='C')
        pdf.cell(50, 10, str(row['tiempo']), border=1, align='C')
        pdf.ln()

    pdf.output(f'Informes/Informe_Top_{x}_Clientes.pdf')
    pdf_filename = f'Informes/Informe_Top_{x}_Clientes.pdf'

    return send_file(pdf_filename, as_attachment=True)

def calculateTopTipos(x):
    con = crearBBDD()
    dataframe = pd.read_sql(
        "SELECT * FROM tickets_emitidos t JOIN contactos_con_empleados c ON c.id_ticket=t.id JOIN tipos_incidentes i ON t.tipo_incidencia=i.id_incidencia",
        con)
    tiempoMaxInci = dataframe.groupby('id_incidencia').agg(
        nombre_incidencia=('nombre', 'first'),
        tiempo=('tiempo', 'sum')
    )

    tiempoMaxInci_ordenado = tiempoMaxInci.sort_values(by='tiempo', ascending=False)[:x]

    return tiempoMaxInci_ordenado

@app.route('/top_tipos/<int:x>')
def get_top_tipos(x):
    tiempoMaxInci_ordenado = calculateTopTipos(x)
    tabla_html = tiempoMaxInci_ordenado.to_html(classes='data')
    return render_template('top_clientes.html', tabla_html=tabla_html)

def obtainLastVulns():
    req = requests.get("https://cve.circl.lu/api/last")
    data = json.loads(req.text)
    ids = []
    descriptions = []
    dates = []
    for doc in data:
        try:
            if len(ids) < 10:
                ids.append(doc['cveMetadata']['cveId'].strip())
                descriptions.append(doc['containers']['cna']['descriptions'][0]['value'].strip())
                dates.append(doc['cveMetadata']['dateUpdated'])
            else:
                break
        except:
            pass

    return ids, descriptions, dates
@app.route('/last10_vulns')
def get_last_vulns():
    ids, descriptions, dates = obtainLastVulns()

    return render_template('last10_vulns.html', cves_ids=ids, cves_descriptions=descriptions, cves_dates=dates)

@app.route('/last10_vulns/downloadPDF')
def generateVulnPDF():
    ids, descriptions, dates = obtainLastVulns()

    pdf = generatePDF()

    pdf.cell(200, 10, txt=f'Informe Top 10 Últimas Vulnerabilidades', ln=True, align='C')
    pdf.ln(10)

    pdf.set_fill_color(200, 200, 200)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, "Nombre Cliente", border=1, align='C', fill=True)
    pdf.cell(50, 10, "Incidencias", border=1, align='C', fill=True)
    pdf.ln()

    for element in range(len(ids)):
        pdf.cell(50, 10, ids[element], border=1, align='C')
        pdf.cell(100, 50, descriptions[element], border=1, align='C')
        pdf.cell(50, 10, dates[element], border=1, align='C')
        pdf.ln()

    pdf.output(f'Informes/Informe_Top_10_Vulnerabilidades.pdf')
    pdf_filename = f'Informes/Informe_Top_10_Vulnerabilidades.pdf'

    return send_file(pdf_filename, as_attachment=True)
@app.route('/news')
def get_latest_cybersecurity_news():
    api_key = '3dc2316e4020483398ca6152bd8a7aa4'
    url = f'https://newsapi.org/v2/everything?q=cybersecurity&sortBy=publishedAt&apiKey={api_key}'
    response = requests.get(url)
    news_data = response.json()

    articles = []
    if news_data['status'] == 'ok':
        for article in news_data['articles']:
            articles.append({
                'title': article['title'],
                'description': article['description'],
                'url': article['url'],
                'publishedAt': article['publishedAt']
            })

    return render_template('latest_news.html', articles=articles)

@app.route('/top_tipos/<int:x>/downloadPDF')
def generateTopTiposPDF(x):
    tiempoMaxInci_ordenado = calculateTopTipos(x)

    pdf = generatePDF()

    pdf.cell(200, 10, txt=f'Informe Top {x} Tipos de incidencia con mayor tiempo de resolucion', ln=True, align='C')
    pdf.ln(10)

    pdf.set_fill_color(200, 200, 200)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, "Nombre Incidencia", border=1, align='C', fill=True)
    pdf.cell(50, 10, "Tiempo", border=1, align='C', fill=True)
    pdf.ln()

    for index, row in tiempoMaxInci_ordenado.iterrows():
        pdf.cell(100, 10, row['nombre_incidencia'], border=1, align='C')
        pdf.cell(50, 10, str(row['tiempo']), border=1, align='C')
        pdf.ln()

    pdf.output(f'Informes/Informe_Top_{x}_Incidencia.pdf')
    pdf_filename = f'Informes/Informe_Top_{x}_Incidencia.pdf'

    return send_file(pdf_filename, as_attachment=True)

def generatePDF():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    return pdf

if __name__ == '__main__':
    app.run(debug=False)
