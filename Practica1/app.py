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

def calcular_metricas(con):

    dataFrameMI5 = pd.read_sql("SELECT satisfaccion_cliente FROM tickets_emitidos", con)
    media = round(dataFrameMI5.mean().values[0], 3)
    desviacion_estandar = round(dataFrameMI5.std().values[0], 3)

    #Media y desviación estándar del total del número de incidentes por cliente.
    incidentes_por_cliente = pd.read_sql("SELECT cliente, COUNT(*) AS num_incidentes FROM tickets_emitidos GROUP BY cliente", con)
    media_incidentes = round(incidentes_por_cliente['num_incidentes'].mean(), 3)
    desviacion_estandar_incidentes = round(incidentes_por_cliente['num_incidentes'].std(), 3)

    dataframeHoras = pd.read_sql("SELECT t.id, c.tiempo FROM tickets_emitidos t JOIN contactos_con_empleados c ON t.id = c.id_ticket", con)
    horasIncidente = dataframeHoras.groupby('id').agg(
        media=('tiempo', 'mean'),
        desviacion=('tiempo', 'std')
    ).reset_index()

    dataframeHorasEmp = pd.read_sql("SELECT id_emp, tiempo FROM contactos_con_empleados", con)
    horasRealizadas = dataframeHorasEmp.groupby('id_emp').agg(
        total_horas = ('tiempo', 'sum')
    )

    dataframeTickets = pd.read_sql("SELECT id, fecha_apertura, fecha_cierre FROM tickets_emitidos", con)
    # Convertir las columnas de fechas a datetime
    dataframeTickets['fecha_apertura'] = pd.to_datetime(dataframeTickets['fecha_apertura'])
    dataframeTickets['fecha_cierre'] = pd.to_datetime(dataframeTickets['fecha_cierre'])
    # Calcular la diferencia en días
    dataframeTickets['dias_diferencia'] = (dataframeTickets['fecha_cierre'] - dataframeTickets['fecha_apertura']).dt.days

    dataframeIncidentes = pd.read_sql("SELECT id_emp, id_ticket FROM contactos_con_empleados", con)
    dataframeIncidentes = dataframeIncidentes.groupby('id_emp').agg(
        tickets_atendidos = ('id_emp', 'count')
    )
    id_emp_max = dataframeIncidentes['tickets_atendidos'].idxmax()
    id_emp_min = dataframeIncidentes['tickets_atendidos'].idxmin()

    return media, desviacion_estandar, media_incidentes, desviacion_estandar_incidentes, horasIncidente.to_dict(orient='records'), max(horasRealizadas['total_horas']), min(horasRealizadas['total_horas']), max(dataframeTickets['dias_diferencia']), min(dataframeTickets['dias_diferencia']), id_emp_max, max(dataframeIncidentes['tickets_atendidos']), id_emp_min, min(dataframeIncidentes['tickets_atendidos'])

def calcular_estadisticas(df, columna):
    return df[columna].agg(['mean', 'median', 'var', 'max', 'min'])

def calcularAgrupaciones(con):
    dataFrameConjunto = pd.read_sql(
        "SELECT t.id, t.fecha_apertura, t.fecha_cierre, i.*, c.id_cli, c.nombre AS nombre_cli, e.id_emp, e.nombre AS nombre_emp, e.nivel FROM tickets_emitidos t JOIN tipos_incidentes i ON t.tipo_incidencia = i.id_incidencia JOIN contactos_con_empleados ce ON t.id = ce.id_ticket JOIN clientes c ON t.cliente = c.id_cli JOIN empleados e ON ce.id_emp = e.id_emp",
        con)
    dataFrameConjuntoFraude = dataFrameConjunto[dataFrameConjunto["nombre"] == "Fraude"].copy()
    dataFrameConjuntoFraude["fecha_apertura"] = pd.to_datetime(
        dataFrameConjuntoFraude["fecha_apertura"])  # Transforma la fecha de apertura a formato datetime
    dataFrameConjuntoFraude["dia_apertura"] = dataFrameConjuntoFraude[
        "fecha_apertura"].dt.dayofweek  # Se saca el día de apertura

    # ---------------- INCIDENTES ----------------
    # 1. Número de incidentes por empleado
    incidentes_por_empleado = dataFrameConjuntoFraude.groupby("id_emp").agg(
        numero_incidentes=('id', 'nunique')).reset_index()
    # 2. Número de incidentes por empleados con nivel entre 1 y 3
    incidentes_por_nivel = dataFrameConjuntoFraude[dataFrameConjuntoFraude["nivel"].between(1, 3)].groupby(
        "id_emp").agg(numero_incidentes=('id_emp', 'count')).reset_index()
    # 3. Número de incidentes por cliente
    incidentes_por_cliente = dataFrameConjuntoFraude.groupby("id_cli").agg(
        numero_incidentes=('id', 'nunique')).reset_index()
    # 4. Número de incidentes por tipo de incidente
    incidentes_por_tipo = dataFrameConjunto.groupby("id_incidencia").agg(nombre_incidencia=('nombre', 'first'),
                                                                         numero_incidentes=(
                                                                         'id', 'nunique')).reset_index()
    # 5. Número de incidentes por día de la semana
    incidentes_por_dia = dataFrameConjuntoFraude.groupby("dia_apertura").agg(
        numero_incidentes=('dia_apertura', 'count')).reset_index()

    estadisticas_empleado = calcular_estadisticas(incidentes_por_empleado, 'numero_incidentes')
    estadisticas_nivel = calcular_estadisticas(incidentes_por_nivel, 'numero_incidentes')
    estadisticas_tipo = calcular_estadisticas(incidentes_por_tipo, 'numero_incidentes')
    estadisticas_cliente = calcular_estadisticas(incidentes_por_cliente, 'numero_incidentes')
    estadisticas_dia = calcular_estadisticas(incidentes_por_dia, 'numero_incidentes')



    # ---------------- ACTUACIONES ----------------
    actuaciones_por_empleado = dataFrameConjuntoFraude.groupby("id_emp").agg(
        actuaciones=('id_emp', 'count')).reset_index()
    actuaciones_por_nivel = dataFrameConjuntoFraude[dataFrameConjuntoFraude["nivel"].between(1, 3)].groupby(
        "id_emp").agg(actuaciones=('id_emp', 'count')).reset_index()
    actuaciones_por_tipo = dataFrameConjunto.groupby("id_incidencia").agg(nombre_incidencia=('nombre', 'first'),
                                                                          actuaciones=('id_emp', 'count')).reset_index()
    actuaciones_por_cliente = dataFrameConjuntoFraude.groupby("id_cli").agg(
        actuaciones=('id_emp', 'count')).reset_index()
    actuaciones_por_dia = dataFrameConjuntoFraude.groupby("dia_apertura").agg(
        actuaciones=('id_emp', 'count')).reset_index()

    estadisticas_empleado_act = calcular_estadisticas(actuaciones_por_empleado, 'actuaciones')
    estadisticas_nivel_act = calcular_estadisticas(actuaciones_por_nivel, 'actuaciones')
    estadisticas_tipo_act = calcular_estadisticas(actuaciones_por_tipo, 'actuaciones')
    estadisticas_cliente_act = calcular_estadisticas(actuaciones_por_cliente, 'actuaciones')
    estadisticas_dia_act = calcular_estadisticas(actuaciones_por_dia, 'actuaciones')

    return incidentes_por_empleado.to_dict(orient='records'), estadisticas_empleado.to_dict(), incidentes_por_nivel.to_dict(orient='records'), incidentes_por_tipo.to_dict(orient='records'), estadisticas_tipo.to_dict(), incidentes_por_cliente.to_dict(orient='records'), estadisticas_cliente.to_dict(), incidentes_por_dia.to_dict(orient='records'), estadisticas_dia.to_dict(), actuaciones_por_empleado.to_dict(orient='records'), estadisticas_empleado_act.to_dict(), actuaciones_por_nivel.to_dict(orient='records'), estadisticas_empleado_act.to_dict(), actuaciones_por_tipo.to_dict(orient='records'), estadisticas_tipo_act.to_dict(), actuaciones_por_cliente.to_dict(orient='records'), estadisticas_cliente_act.to_dict(), actuaciones_por_dia.to_dict(orient='records'), estadisticas_dia_act.to_dict()

def generar_graficas(con):
    # Crear directorio para imágenes
    os.makedirs('static/img', exist_ok=True)
    paths = {}

    # Gráfico 1: Tiempo medio de resolución
    df = pd.read_sql("SELECT fecha_apertura, fecha_cierre, es_mantenimiento FROM tickets_emitidos", con)
    df['tiempo_resolucion'] = (pd.to_datetime(df['fecha_cierre']) - pd.to_datetime(df['fecha_apertura'])).dt.days
    media_tiempos = df.groupby('es_mantenimiento')['tiempo_resolucion'].mean()
    
    plt.figure(figsize=(6, 4))
    media_tiempos.plot(kind='bar', color=['blue', 'orange'])
    plt.title('Tiempo Medio de Resolución por Tipo')
    plt.xticks([0, 1], ['No Mantenimiento', 'Mantenimiento'], rotation=0)
    plt.savefig('static/img/grafica1.png')
    plt.close()
    paths['grafica1'] = 'img/grafica1.png'

    # Gráfico 2: Distribución tiempos
    df = pd.read_sql("SELECT fecha_apertura, fecha_cierre, tipo_incidencia FROM tickets_emitidos", con)
    df['tiempo_resolucion'] = (pd.to_datetime(df['fecha_cierre']) - pd.to_datetime(df['fecha_apertura'])).dt.days
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['tipo_incidencia'].astype(str), y=df['tiempo_resolucion'], showfliers=False)
    plt.title('Distribución del Tiempo de Resolución')
    plt.savefig('static/img/grafica2.png')
    plt.close()
    paths['grafica2'] = 'img/grafica2.png'

    # Gráfico 3: Clientes más críticos
    df = pd.read_sql(
        "SELECT cliente, COUNT(*) AS num_incidentes FROM tickets_emitidos WHERE es_mantenimiento = 1 AND tipo_incidencia != 1 GROUP BY cliente ORDER BY num_incidentes DESC LIMIT 5",
        con)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='cliente', y='num_incidentes', data=df, palette='Reds')
    plt.title('Top 5 Clientes Más Críticos')
    plt.savefig('static/img/grafica3.png')
    plt.close()
    paths['grafica3'] = 'img/grafica3.png'

    # Gráfico 4: Actuaciones por cliente
    df = pd.read_sql("""
        SELECT t.cliente AS id_cliente, 
               COUNT(*) as total_actuaciones 
        FROM contactos_con_empleados c
        JOIN tickets_emitidos t ON c.id_ticket = t.id
        GROUP BY t.cliente
    """, con)
    plt.figure(figsize=(10, 6))
    plt.bar(df['id_cliente'].astype(str), df['total_actuaciones'])
    plt.title('Actuaciones por Cliente')
    plt.savefig('static/img/grafica4.png')
    plt.close()
    paths['grafica4'] = 'img/grafica4.png'

    # Gráfico 5: Actuaciones por día
    df = pd.read_sql("SELECT fecha FROM contactos_con_empleados", con)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['dia_semana'] = df['fecha'].dt.day_name()
    dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    counts = df['dia_semana'].value_counts().reindex(dias_orden, fill_value=0)

    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.values)
    plt.title('Actuaciones por Día de la Semana')
    plt.xticks(rotation=45)
    plt.savefig('static/img/grafica5.png')
    plt.close()
    paths['grafica5'] = 'img/grafica5.png'

    return paths

@app.route('/')
def home():
    con = crearBBDD()
    cur = con.cursor()
    img_paths = generar_graficas(con)
    dataFrameClientes = pd.DataFrame(cur.execute("SELECT * FROM clientes"))
    dataFrameEmpleados = pd.DataFrame(cur.execute("SELECT * FROM empleados"))
    dataFrameTicketsEmitidos = pd.DataFrame(cur.execute("SELECT * FROM tickets_emitidos"))
    dataFrameTiposIncidentes = pd.DataFrame(cur.execute("SELECT * FROM tipos_incidentes"))

    (media, desviacion_estandar, media_incidentes, desviacion_estandar_incidentes,
     horasxIncidente, maxHorasEmpleado, minHorasEmpleado, maxDiasIncidente,
     minDiasIncidente, id_emp_max, maxAtendidos, id_emp_min, minAtendidos) = calcular_metricas(con)

    (incidentes_por_empleado, estadisticas_empleado, incidentes_por_nivel, incidentes_por_tipo, estadisticas_tipo,
     incidentes_por_cliente, estadisticas_cliente, incidentes_por_dia, estadisticas_dia, actuaciones_por_empleado,
     estadisticas_empleado_act, actuaciones_por_nivel, estadisticas_empleado_act, actuaciones_por_tipo, estadisticas_tipo_act, actuaciones_por_cliente,
     estadisticas_cliente_act, actuaciones_por_dia, estadisticas_dia_act) = calcularAgrupaciones(con)

    return render_template("index.html",
                           img_paths=img_paths,
                           nclientes=len(dataFrameClientes), nempleados=len(dataFrameEmpleados),
                           ntickets=str(len(dataFrameTicketsEmitidos)), ntipos=str(len(dataFrameTiposIncidentes)), media = media,
                           desviacion_estandar = desviacion_estandar, media_incidentes=  media_incidentes, desviacion_estandar_incidentes= desviacion_estandar_incidentes,
                           horasxIncidente = horasxIncidente, maxHorasEmpleado = maxHorasEmpleado, minHorasEmpleado = minHorasEmpleado, maxDiasIncidente = maxDiasIncidente,
                           minDiasIncidente = minDiasIncidente, id_emp_max = id_emp_max, maxAtendidos = maxAtendidos, id_emp_min = id_emp_min, minAtendidos = minAtendidos,
                           incidentes_por_empleado = incidentes_por_empleado, estadisticas_empleado=estadisticas_empleado, incidentes_por_nivel = incidentes_por_nivel,
                           incidentes_por_tipo = incidentes_por_tipo, actuaciones_por_tipo = actuaciones_por_tipo, estadisticas_tipo = estadisticas_tipo, incidentes_por_cliente = incidentes_por_cliente,
                           estadisticas_cliente = estadisticas_cliente, incidentes_por_dia = incidentes_por_dia, estadisticas_dia = estadisticas_dia,
                           actuaciones_por_empleado = actuaciones_por_empleado, estadisticas_empleado_act = estadisticas_empleado_act, actuaciones_por_nivel=actuaciones_por_nivel,
                           estadisticas_tipo_act = estadisticas_tipo_act, actuaciones_por_cliente = actuaciones_por_cliente, estadisticas_cliente_act = estadisticas_cliente_act, actuaciones_por_dia = actuaciones_por_dia, estadisticas_dia_act=estadisticas_dia_act)

if __name__ == '__main__':
    app.run(debug=False)
