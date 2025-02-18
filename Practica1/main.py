import sqlite3
import json
def leerJSON(cur, con):
    ficheroJSON = open('datos.json', 'r')
    datos = json.load(ficheroJSON)

    for elemento in datos["clientes"]:
        contador = 1

        cur.execute("INSERT OR IGNORE INTO clientes (id_cli, nombre, provincia, telefono)"
                "VALUES ('%d', '%s', '%s', '%s')" % (int(elemento['id_cli']), elemento['nombre'], elemento['provincia'], elemento['telefono']))
        con.commit()
        contador += 1


    for elemento in datos["tickets_emitidos"]:
        cur.execute("INSERT OR IGNORE INTO tickets_emitidos (cliente, fecha_apertura, fecha_cierre, es_mantenimiento, satisfaccion_cliente, tipo_incidencia)"
                    "VALUES ('%s', '%s', '%s', '%d', '%d', '%d')" % (elemento['cliente'], elemento['fecha_apertura'], elemento['fecha_cierre'], int(elemento['es_mantenimiento']), int(elemento['satisfaccion_cliente']), int(elemento['tipo_incidencia'])))
        con.commit()

    for elemento in datos["empleados"]:
        cur.execute("INSERT OR IGNORE INTO empleados (fecha_contrato, id_emp, nivel, nombre)"
                    "VALUES ('%s', '%d', '%d', '%s')" % (elemento['fecha_contrato'], int(elemento['id_emp']), int(elemento['nivel']), elemento['nombre']))
        con.commit()


    for elemento in datos["tipos_incidentes"]:
        cur.execute("INSERT OR IGNORE INTO tipos_incidentes (id_cli, nombre)"
                    "VALUES ('%d', '%s')" % (int(elemento['id_cli']), elemento['nombre']))
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
                "id_cli INTEGER,"
                "nombre TEXT "
                ");")
    con.commit()

    leerJSON(cur, con)

crearBBDD()




