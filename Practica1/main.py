import sqlite3
import json
def leerJSON():
    ficheroJSON = open('datos.json', 'r')
    datos = json.load(ficheroJSON)
    print()
    #print(datos)
    #print(datos["tickets_emitidos"])
    #print(datos["tipos_incidentes"])


def crearBBDD():
    con = sqlite3.connect('database.db')
    cur = con.cursor()


    #cur.execute("CREATE TABLE IF NOT EXISTS tickets_emitidos (""")


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
                "nombre TEXT PRIMARY KEY"
                ");")

    cur.execute("INSERT INTO tipos_incidentes(id_cli, nombre) VALUES ('1', 'Manolo')")

    con.commit()
    con.close()

leerJSON()
crearBBDD()


