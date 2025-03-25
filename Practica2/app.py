import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from flask import Flask, render_template
app = Flask(__name__)

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

@app.route('/')
def home():

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=False)
