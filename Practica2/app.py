import sqlite3
import json
from datetime import datetime

import pandas as pd
import requests
from sklearn import tree
import graphviz
import numpy as np
import os
from flask import Flask, render_template, jsonify, send_file, request
from fpdf import FPDF
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
from reportlab.lib import colors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
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

    tabla_html = cliMaxIncidents_ordenado.to_html(index=False, classes='data')
    return render_template('top_clientes.html', tabla_clientes_html=tabla_html)

@app.route('/top_clientes/<int:x>/downloadPDF')
def generateTopClientesPDF(x):
    cliMaxIncidents_ordenado = calculateTopClientes(x)

    os.makedirs('Informes', exist_ok=True)
    pdf = generatePDF()

    pdf.cell(200, 10, txt=f'Informe Top {x} Clientes con Incidencias', ln=True, align='C')
    pdf.ln(10)

    pdf.set_fill_color(200, 200, 200)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, "Nombre Cliente", border=1, align='C', fill=True)
    pdf.cell(50, 10, "Incidencias", border=1, align='C', fill=True)
    pdf.ln()

    for index, row in cliMaxIncidents_ordenado.iterrows():
        pdf.cell(100, 10, row['nombre_cliente'], border=1, align='C')
        pdf.cell(50, 10, str(row['incidencias']), border=1, align='C')
        pdf.ln()

    pdf_filename = f'Informes/Informe_Top_{x}_Clientes.pdf'
    pdf.output(pdf_filename)

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
    tabla_html = tiempoMaxInci_ordenado.to_html(index=False, classes='data')
    return render_template('top_tipos.html', tabla_tipo_html=tabla_html)

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


# ... (resto de tus importaciones y código de Flask) ...

@app.route('/last10_vulns/downloadPDF')
def generateVulnPDF():
    try:
        ids, descriptions, dates = obtainLastVulns()

        pdf_filename = 'Informes/Informe_Top_10_Vulnerabilidades.pdf'
        os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)

        pdf = SimpleDocTemplate(pdf_filename, pagesize=letter)
        elements = []

        styles = getSampleStyleSheet()

        title_style = styles['h1']
        title_style.alignment = TA_CENTER

        header_style = styles['Normal']
        header_style.textColor = colors.white
        header_style.fontName = 'Helvetica-Bold'
        header_style.alignment = TA_CENTER # Centrar texto de cabecera


        normal_style = styles['Normal']
        normal_style.textColor = colors.black
        normal_style.alignment = TA_LEFT
        normal_style.fontSize = 9
        normal_style.leading = 11

        # --- TÍTULO ---
        title_text = "Top 10 vulnerabilidades"
        title = Paragraph(title_text, title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))

        # --- DATOS PARA LA TABLA ---
        data = [[Paragraph("ID CVE", header_style),
                 Paragraph("Descripción", header_style),
                 Paragraph("Fecha", header_style)]]

        for i in range(len(ids)):
            desc_paragraph = Paragraph(descriptions[i] if descriptions[i] else '', normal_style)
            id_paragraph = Paragraph(ids[i] if ids[i] else '', normal_style)
            date_paragraph = Paragraph(dates[i] if dates[i] else '', normal_style)
            data.append([id_paragraph, desc_paragraph, date_paragraph])

        # --- TABLA ---
        table = Table(data, colWidths=[90, 320, 90])

        # --- ESTILO DE LA TABLA ---
        style = TableStyle([
            # Cabecera: Fondo #4CAF50, texto centrado (controlado por Paragraph)
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),

            # --- MODIFICACIÓN AQUÍ: Datos ---
            # Fondo negro, alineación vertical arriba
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('VALIGN', (0, 1), (-1, -1), 'TOP'),
             ('ALIGN', (0, 1), (0, -1), 'LEFT'), # Columna ID
             ('ALIGN', (1, 1), (1, -1), 'LEFT'), # Columna Descripción
             ('ALIGN', (2, 1), (2, -1), 'CENTER'), # Columna Fecha (Ejemplo centrado)

            # Rejilla y Relleno
            ('GRID', (0, 0), (-1, -1), 1, colors.black), # Rejilla negra (podría no verse sobre fondo negro)
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ])
        table.setStyle(style)
        elements.append(table)
        pdf.build(elements)
        return send_file(pdf_filename, as_attachment=True)

    except Exception as e:
        print(f"Error generando PDF: {e}")
        # import traceback
        # print(traceback.format_exc())
        return "Error al generar el informe PDF", 500


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

    pdf_filename = f'Informes/Informe_Top_{x}_Tipos.pdf'
    pdf.output(pdf_filename)
    

    return send_file(pdf_filename, as_attachment=True)

def generatePDF():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    return pdf


@app.route('/decisionTree')
def makeDecisionTreeClasifier():
    dataClasified = open('data_clasified.json', 'r')
    dataClasified = json.load(dataClasified)

    tickets = dataClasified['tickets_emitidos']

    X = []
    y = []

    for ticket in tickets:
        features = [
            ticket['tipo_incidencia'],
            ticket['es_mantenimiento'],
            ticket['satisfaccion_cliente']
        ]
        X.append(features)

        if ticket['es_critico']:
            y.append(1)
        else:
            y.append(0)

    X = np.array(X)
    y = np.array(y)

    feature_names = ['Tipo Incidencia', 'Es Mantenimiento', 'Satisfacción Cliente']
    class_names = ['No Crítico', 'Crítico']

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True, rounded=True,
                                    special_characters=True, proportion=True,impurity=False)


    #REVISAR
    graph = graphviz.Source(dot_data)

    output_path = 'static/graphs/decision_tree_critico'
    graph.render(output_path, format='png', cleanup=True)

    return send_file(output_path + '.png', mimetype='image/png')

# Función para calcular la duración del ticket en días
def calcular_duracion(fecha_apertura, fecha_cierre):
    apertura = datetime.strptime(fecha_apertura, "%Y-%m-%d")
    cierre = datetime.strptime(fecha_cierre, "%Y-%m-%d")
    return (cierre - apertura).days

    # Función para calcular el tiempo total de contacto con empleados
def calcular_tiempo_contacto(contactos):
    return sum(contacto['tiempo'] for contacto in contactos)

@app.route('/predictions')
def predictions():
    return render_template('predictions.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos del formulario
    fecha_apertura = request.form['fecha_apertura']
    fecha_cierre = request.form['fecha_cierre']
    es_mantenimiento = request.form['es_mantenimiento'] == 'true'
    satisfaccion = int(request.form['satisfaccion_cliente'])
    tipo_incidencia = int(request.form['tipo_incidencia'])
    metodo = request.form['metodo']  # Corregí un typo aquí (metodo)
    tiempo_contacto = float(request.form.get('tiempo_contacto', 2.0))
    num_contactos = int(request.form.get('num_contactos', 1))

    if metodo == "regresion":
        with open('data_clasified.json', 'r') as file:
            data = json.load(file)

        tickets = data['tickets_emitidos']

        # Preparar los datos para el modelo
        features = []
        labels = []

        for ticket in tickets:
            duracion = calcular_duracion(ticket['fecha_apertura'], ticket['fecha_cierre'])
            ticket_tiempo_contacto = calcular_tiempo_contacto(ticket['contactos_con_empleados'])
            ticket_num_contactos = len(ticket['contactos_con_empleados'])

            ticket_es_mantenimiento = 1 if ticket['es_mantenimiento'] else 0
            ticket_tipo_incidencia = ticket['tipo_incidencia']
            ticket_satisfaccion = ticket['satisfaccion_cliente']

            # Agregar características
            features.append([
                duracion,
                ticket_tiempo_contacto,
                ticket_num_contactos,
                ticket_es_mantenimiento,
                ticket_tipo_incidencia,
                ticket_satisfaccion
            ])

            # Etiqueta (variable objetivo)
            labels.append(1 if ticket['es_critico'] else 0)

        df = pd.DataFrame(features, columns=[
            'duracion',
            'tiempo_contacto',
            'num_contactos',
            'es_mantenimiento',
            'tipo_incidencia',
            'satisfaccion'
        ])
        df['es_critico'] = labels

        # Dividir datos en entrenamiento y prueba
        X = df.drop('es_critico', axis=1)
        y = df['es_critico']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Estandarizar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Entrenar el modelo
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        # Evaluar el modelo
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precisión del modelo: {accuracy:.2f}")

        # Calcular duración para la predicción
        duracion_pred = calcular_duracion(fecha_apertura, fecha_cierre)

        # Preparar características para predicción usando valores del formulario
        features_pred = [
            duracion_pred,
            tiempo_contacto,
            num_contactos,
            1 if es_mantenimiento else 0,
            tipo_incidencia,
            satisfaccion
        ]

        # Estandarizar y predecir
        features_scaled = scaler.transform([features_pred])
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)[0]

        es_critico = bool(prediction[0])
        probabilidad = prediction_proba[1] if es_critico else prediction_proba[0]

        return jsonify({
            'es_critico': es_critico,
            'probabilidad': float(probabilidad),
            'mensaje': "Crítico" if es_critico else "No crítico",
            'metodo': "Regresión Logística"
        })

    elif metodo == "decision":
        return jsonify({
            'es_critico': '0',
            'mensaje': "Crítico" if 0 else "No crítico",
            'metodo': 'Decision Tree'
        })

    elif metodo == "random":
        return jsonify({
            'es_critico': '0',
            'mensaje': "Crítico" if 0 else "No crítico",
            'metodo': 'Random Forest'
        })


if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=False)
