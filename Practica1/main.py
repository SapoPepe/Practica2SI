import sqlite3
import json
def leerJSON(cur, con):
    ficheroJSON = open('datos.json', 'r')
    datos = json.load(ficheroJSON)

    for elemento in datos["clientes"]:
        contador = 1
        clave = list(elemento.keys())[0]
        print(clave)
        cur.execute("INSERT INTO clientes (id_cli, nombre, provincia, telefono)"
                "VALUES ('%d', '%s', '%s', '%s')" % (int(elemento['id_cli']), elemento['nombre'], elemento['provincia'], elemento['telefono']))
        con.commit()
        contador += 1







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
                "id_emp INTEGER PRIMARY KEY,"
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




