import sqlite3
import json

ficheroJSON = open('datos.json', 'r')
datos = json.load(ficheroJSON)
print(datos)
print("Fichero Leido")

