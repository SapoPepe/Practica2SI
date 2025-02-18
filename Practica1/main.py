import sqlite3
import json
import pandas as pd
def leerJSON(cur, con):
    ficheroJSON = open('datos.json', 'r')
    datos = json.load(ficheroJSON)

    for elemento in datos["clientes"]:
        cur.execute("INSERT OR IGNORE INTO clientes (id_cli, nombre, provincia, telefono)"
                "VALUES ('%d', '%s', '%s', '%s')" % (int(elemento['id_cli']), elemento['nombre'], elemento['provincia'], elemento['telefono']))
        con.commit()

    for elemento in datos["tickets_emitidos"]:
        cur.execute("INSERT OR IGNORE INTO tickets_emitidos (cliente, fecha_apertura, fecha_cierre, es_mantenimiento, satisfaccion_cliente, tipo_incidencia)"
                    "VALUES ('%s', '%s', '%s', '%d', '%d', '%d')" % (elemento['cliente'], elemento['fecha_apertura'], elemento['fecha_cierre'], int(elemento['es_mantenimiento']), int(elemento['satisfaccion_cliente']), int(elemento['tipo_incidencia'])))
        con.commit()

    for elemento in datos["empleados"]:
        cur.execute("INSERT OR IGNORE INTO empleados (fecha_contrato, id_emp, nivel, nombre)"
                    "VALUES ('%s', '%d', '%d', '%s')" % (elemento['fecha_contrato'], int(elemento['id_emp']), int(elemento['nivel']), elemento['nombre']))
        con.commit()


    for elemento in datos["tipos_incidentes"]:
        cur.execute("INSERT OR IGNORE INTO tipos_incidentes (id_incidencia, nombre)"
                    "VALUES ('%d', '%s')" % (int(elemento['id_incidencia']), elemento['nombre']))
        con.commit()







def crearBBDD():
    con = sqlite3.connect('database.db')
    cur = con.cursor()


    cur.execute("CREATE TABLE IF NOT EXISTS tickets_emitidos ("
                "cliente TEXT, "
                "fecha_apertura TEXT, "
                "fecha_cierre TEXT, "
                "es_mantenimiento INTEGER, "
                "satisfaccion_cliente INTEGER, "
                "tipo_incidencia INTEGER );")


    cur.execute("CREATE TABLE IF NOT EXISTS clientes ("
                "id_cli INTEGER,"
                "nombre TEXT,"
                "provincia TEXT,"
                "telefono TEXT"
                ");")

    cur.execute("CREATE TABLE IF NOT EXISTS empleados ("
                "fecha_contrato TEXT,"
                "id_emp INTEGER,"
                "nivel INTEGER,"
                "nombre TEXT"
                ");")

    cur.execute("CREATE TABLE IF NOT EXISTS tipos_incidentes ("
                "id_incidencia INTEGER,"
                "nombre TEXT "
                ");")
    con.commit()

    leerJSON(cur, con)


    print("Numero de Muestras:")
    dataFrameClientes = pd.DataFrame(cur.execute("SELECT * FROM clientes"))
    print(dataFrameClientes)
    dataFrameEmpleados = pd.DataFrame(cur.execute("SELECT * FROM empleados"))
    dataFrameTicketsEmitidos = pd.DataFrame(cur.execute("SELECT * FROM tickets_emitidos"))
    dataFrameTiposIncidentes = pd.DataFrame(cur.execute("SELECT * FROM tipos_incidentes"))
    print("Numero de Clientes: " + str(len(dataFrameClientes)))
    print("Numero de Empleados: " + str(len(dataFrameEmpleados)))
    print("Numero de Tickets Emitidos: " + str(len(dataFrameTicketsEmitidos)))
    print("Numero de Tipos de Incidente: " + str(len(dataFrameTiposIncidentes)))
    return con

def calcular_metricas(con):
    query = "SELECT * FROM tickets_emitidos"
    tickets = pd.read_sql(query, con)
    print("\nCalcular metricas\n")
    print(tickets)
    # 1
    total_muestras = len(tickets)
    print(f"Total de muestras: {total_muestras}")

    # 2
    incidentes_alta_valoracion = tickets[tickets['satisfaccion_cliente'] >= 5]
    media_valoracion = incidentes_alta_valoracion['satisfaccion_cliente'].mean()
    std_valoracion = incidentes_alta_valoracion['satisfaccion_cliente'].std()
    print(f"Media valoración >=5: {media_valoracion:.2f} ± {std_valoracion:.2f}")

con = crearBBDD()
calcular_metricas(con)

