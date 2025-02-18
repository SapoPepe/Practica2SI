import sqlite3

import json

f=open('ejemplo.json','r')
datos=json.load(f)
print(datos)
print(datos["fichajes"])


con = sqlite3.connect('datos.db')
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS fichajes ("
    "id INTEGER PRIMARY KEY,"
    "nombre TEXT,"
    "sucursal TEXT,"
    "departamento TEXT"
    ");")
con.commit()
for elem in datos["fichajes"]:
    #print (elem)
    clave= list(elem.keys())[0]
    print(clave)
    print (elem[clave]["nombre"])

    cur.execute("INSERT OR IGNORE INTO fichajes(id,nombre,sucursal,departamento)"\
                "VALUES ('%d','%s', '%s', '%s')" %
                (int(clave), elem[clave]['nombre'], elem[clave]['sucursal'],elem[clave]['departamento']))
    con.commit()
