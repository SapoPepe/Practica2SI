import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

    cur.execute("DROP TABLE clientes;")
    cur.execute("DROP TABLE empleados;")
    cur.execute("DROP TABLE tipos_incidentes;")
    cur.execute("DROP TABLE tickets_emitidos;")
    cur.execute("DROP TABLE contactos_con_empleados;")

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

    print("---------------- Numero de Muestras ----------------")
    dataFrameClientes = pd.DataFrame(cur.execute("SELECT * FROM clientes"))
    dataFrameEmpleados = pd.DataFrame(cur.execute("SELECT * FROM empleados"))
    dataFrameTicketsEmitidos = pd.DataFrame(cur.execute("SELECT * FROM tickets_emitidos"))
    dataFrameTiposIncidentes = pd.DataFrame(cur.execute("SELECT * FROM tipos_incidentes"))
    print("Numero de Clientes: " + str(len(dataFrameClientes)))
    print("Numero de Empleados: " + str(len(dataFrameEmpleados)))
    print("Numero de Tickets Emitidos: " + str(len(dataFrameTicketsEmitidos)))
    print("Numero de Tipos de Incidente: " + str(len(dataFrameTiposIncidentes)))
    return con

def calcular_metricas(con):
    print("\n///////////////////////////////// Calcular metricas /////////////////////////////////\n")

    print("\n---------------- Valoración >= 5 ----------------")
    dataFrameMI5 = pd.read_sql("SELECT satisfaccion_cliente FROM tickets_emitidos", con)
    media = round(dataFrameMI5.mean().values[0], 3)
    desviacion_estandar = round(dataFrameMI5.std().values[0], 3)
    print(f"Media: {media}\nDesviación estandar: {desviacion_estandar}")

    print("\n---------------- Nº incidentes por cliente ----------------")
    #Media y desviación estándar del total del número de incidentes por cliente.
    incidentes_por_cliente = pd.read_sql("SELECT cliente, COUNT(*) AS num_incidentes FROM tickets_emitidos GROUP BY cliente", con)
    media_incidentes = round(incidentes_por_cliente['num_incidentes'].mean(), 3)
    desviacion_estandar_incidentes = round(incidentes_por_cliente['num_incidentes'].std(), 3)
    print(f"Media: {media_incidentes}\nDesviación estandar: {desviacion_estandar_incidentes}")


    print("\n---------------- Horas totales por tipo de incidente ----------------")
    dataframeHoras = pd.read_sql("SELECT t.id, c.tiempo FROM tickets_emitidos t JOIN contactos_con_empleados c ON t.id = c.id_ticket", con)
    resultado = dataframeHoras.groupby('id').agg(
        media=('tiempo', 'mean'),
        desviacion=('tiempo', 'std')
    ).reset_index()
    print(resultado)


    print("\n---------------- Min y Max del total de horas realizadas por los empleados ----------------")
    dataframeHorasEmp = pd.read_sql("SELECT id_emp, tiempo FROM contactos_con_empleados", con)
    resultado = dataframeHorasEmp.groupby('id_emp').agg(
        total_horas = ('tiempo', 'sum')
    )
    print(f"Max: {max(resultado['total_horas'])}\nMin: {min(resultado['total_horas'])}")



    print("\n---------------- Min y Max del tiempo entre apertura y cierre de ticket ----------------")
    dataframeTickets = pd.read_sql("SELECT id, fecha_apertura, fecha_cierre FROM tickets_emitidos", con)
    # Convertir las columnas de fechas a datetime
    dataframeTickets['fecha_apertura'] = pd.to_datetime(dataframeTickets['fecha_apertura'])
    dataframeTickets['fecha_cierre'] = pd.to_datetime(dataframeTickets['fecha_cierre'])
    # Calcular la diferencia en días
    dataframeTickets['dias_diferencia'] = (dataframeTickets['fecha_cierre'] - dataframeTickets['fecha_apertura']).dt.days

    print(f"Max: {max(dataframeTickets['dias_diferencia'])}\nMin: {min(dataframeTickets['dias_diferencia'])}")



    print("\n---------------- Min y Max del número de incidentes atendidos por cada empleado ----------------")
    dataframeIncidentes = pd.read_sql("SELECT id_emp, id_ticket FROM contactos_con_empleados", con)
    dataframeIncidentes = dataframeIncidentes.groupby('id_emp').agg(
        tickets_atendidos = ('id_emp', 'count')
    )


    id_emp_max = dataframeIncidentes['tickets_atendidos'].idxmax()
    id_emp_min = dataframeIncidentes['tickets_atendidos'].idxmin()

    print(f"Empleado: {id_emp_max} | Max: {max(dataframeIncidentes['tickets_atendidos'])}\nEmpleado: {id_emp_min} | Min: {min(dataframeIncidentes['tickets_atendidos'])}")



def calcular_estadisticas(df, columna):
    return df[columna].agg(['mean', 'median', 'var', 'max', 'min'])



con = crearBBDD()
calcular_metricas(con)


dataFrameConjunto = pd.read_sql("SELECT t.id, t.fecha_apertura, t.fecha_cierre, i.*, c.id_cli, c.nombre AS nombre_cli, e.id_emp, e.nombre AS nombre_emp, e.nivel FROM tickets_emitidos t JOIN tipos_incidentes i ON t.tipo_incidencia = i.id_incidencia JOIN contactos_con_empleados ce ON t.id = ce.id_ticket JOIN clientes c ON t.cliente = c.id_cli JOIN empleados e ON ce.id_emp = e.id_emp", con)
dataFrameConjuntoFraude = dataFrameConjunto[dataFrameConjunto["nombre"] == "Fraude"].copy()
dataFrameConjuntoFraude["fecha_apertura"] = pd.to_datetime(dataFrameConjuntoFraude["fecha_apertura"])   # Transforma la fecha de apertura a formato datetime
dataFrameConjuntoFraude["dia_apertura"] = dataFrameConjuntoFraude["fecha_apertura"].dt.dayofweek    # Se saca el día de apertura

# ---------------- INCIDENTES ----------------
# 1. Número de incidentes por empleado
incidentes_por_empleado = dataFrameConjuntoFraude.groupby("id_emp").agg(numero_incidentes=('id', 'nunique')).reset_index()
# 2. Número de incidentes por empleados con nivel entre 1 y 3
incidentes_por_nivel = dataFrameConjuntoFraude[dataFrameConjuntoFraude["nivel"].between(1, 3)].groupby("id_emp").agg(numero_incidentes=('id_emp', 'count')).reset_index()
# 3. Número de incidentes por cliente
incidentes_por_cliente = dataFrameConjuntoFraude.groupby("id_cli").agg(numero_incidentes=('id', 'nunique')).reset_index()
# 4. Número de incidentes por tipo de incidente
incidentes_por_tipo = dataFrameConjunto.groupby("id_incidencia").agg(nombre_incidencia=('nombre', 'first'), numero_incidentes=('id', 'nunique')).reset_index()
# 5. Número de incidentes por día de la semana
incidentes_por_dia = dataFrameConjuntoFraude.groupby("dia_apertura").agg(numero_incidentes=('dia_apertura', 'count')).reset_index()



estadisticas_empleado = calcular_estadisticas(incidentes_por_empleado, 'numero_incidentes')
estadisticas_nivel = calcular_estadisticas(incidentes_por_nivel, 'numero_incidentes')
estadisticas_tipo = calcular_estadisticas(incidentes_por_tipo, 'numero_incidentes')
estadisticas_cliente = calcular_estadisticas(incidentes_por_cliente, 'numero_incidentes')
estadisticas_dia = calcular_estadisticas(incidentes_por_dia, 'numero_incidentes')

print(incidentes_por_empleado)
print("Estadisticas por empleado:\n", estadisticas_empleado)
print(incidentes_por_nivel)
print("\nEstadisticas por empleado con nivel:\n", estadisticas_empleado)
print(incidentes_por_tipo)
print("\nEstadisticas por tipo de incidente:\n", estadisticas_tipo)
print(incidentes_por_cliente)
print("\nEstadisticas por cliente:\n", estadisticas_cliente)
print(incidentes_por_dia)
print("\nEstadisticas por dia:\n", estadisticas_dia)



# ---------------- ACTUACIONES ----------------
actuaciones_por_empleado = dataFrameConjuntoFraude.groupby("id_emp").agg(actuaciones=('id_emp', 'count')).reset_index()
actuaciones_por_nivel = dataFrameConjuntoFraude[dataFrameConjuntoFraude["nivel"].between(1, 3)].groupby("id_emp").agg(actuaciones=('id_emp', 'count')).reset_index()
actuaciones_por_tipo = dataFrameConjunto.groupby("id_incidencia").agg(nombre_incidencia=('nombre', 'first'), actuaciones=('id_emp', 'count')).reset_index()
actuaciones_por_cliente = dataFrameConjuntoFraude.groupby("id_cli").agg(actuaciones=('id_emp', 'count')).reset_index()
actuaciones_por_dia = dataFrameConjuntoFraude.groupby("dia_apertura").agg(actuaciones=('id_emp', 'count')).reset_index()

estadisticas_empleado_act = calcular_estadisticas(actuaciones_por_empleado, 'actuaciones')
estadisticas_nivel_act = calcular_estadisticas(actuaciones_por_nivel, 'actuaciones')
estadisticas_tipo_act = calcular_estadisticas(actuaciones_por_tipo, 'actuaciones')
estadisticas_cliente_act = calcular_estadisticas(actuaciones_por_cliente, 'actuaciones')
estadisticas_dia_act = calcular_estadisticas(actuaciones_por_dia, 'actuaciones')

print(actuaciones_por_empleado)
print("Estadisticas por empleado:\n", estadisticas_empleado_act)
print(actuaciones_por_nivel)
print("\nEstadisticas por empleado con nivel:\n", estadisticas_empleado_act)
print(actuaciones_por_tipo)
print("\nEstadisticas por tipo de incidente:\n", estadisticas_tipo_act)
print(actuaciones_por_cliente)
print("\nEstadisticas por cliente:\n", estadisticas_cliente_act)
print(actuaciones_por_dia)
print("\nEstadisticas por dia:\n", estadisticas_dia_act)





def generar_graficas(con):
    df = pd.read_sql("SELECT fecha_apertura, fecha_cierre, es_mantenimiento FROM tickets_emitidos", con)
    df['fecha_apertura'] = pd.to_datetime(df['fecha_apertura'])
    df['fecha_cierre'] = pd.to_datetime(df['fecha_cierre'])
    df['tiempo_resolucion'] = (df['fecha_cierre'] - df['fecha_apertura']).dt.days

    media_tiempos = df.groupby('es_mantenimiento')['tiempo_resolucion'].mean()

    plt.figure(figsize=(6, 4))
    media_tiempos.plot(kind='bar', color=['blue', 'orange'])
    plt.xlabel('Tipo de Incidente')
    plt.ylabel('Tiempo Medio de Resolución (días)')
    plt.title('Tiempo Medio de Resolución por Tipo de Incidente')
    plt.xticks(ticks=[0, 1], labels=['No Mantenimiento', 'Mantenimiento'], rotation=0)
    plt.show()

    df = pd.read_sql("SELECT fecha_apertura, fecha_cierre, tipo_incidencia FROM tickets_emitidos", con)
    df['fecha_apertura'] = pd.to_datetime(df['fecha_apertura'])
    df['fecha_cierre'] = pd.to_datetime(df['fecha_cierre'])
    df['tiempo_resolucion'] = (df['fecha_cierre'] - df['fecha_apertura']).dt.days

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['tipo_incidencia'].astype(str), y=df['tiempo_resolucion'], showfliers=False,
                flierprops={'marker': 'o', 'markersize': 5},
                boxprops={'facecolor': 'lightblue'},
                whiskerprops={'linewidth': 2})

    percentil_5 = df['tiempo_resolucion'].quantile(0.05)
    percentil_90 = df['tiempo_resolucion'].quantile(0.90)

    plt.axhline(y=percentil_5, color='red', linestyle='dashed', label='Percentil 5%')
    plt.axhline(y=percentil_90, color='green', linestyle='dashed', label='Percentil 90%')

    plt.xlabel('Tipo de Incidente')
    plt.ylabel('Tiempo de Resolución (días)')
    plt.title('Distribución del Tiempo de Resolución por Tipo de Incidente')
    plt.legend()
    plt.show()

    df = pd.read_sql(
        "SELECT cliente, COUNT(*) AS num_incidentes FROM tickets_emitidos WHERE es_mantenimiento = 1 AND tipo_incidencia != 1 GROUP BY cliente ORDER BY num_incidentes DESC LIMIT 5",
        con)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='cliente', y='num_incidentes', hue='cliente', data=df, palette='Reds', legend=False)
    plt.xlabel('ID Cliente')
    plt.ylabel('Número de Incidentes')
    plt.title('Top 5 Clientes Más Críticos')
    plt.show()

    # Gráfico de actuaciones por empleado
    df = pd.read_sql("""
        SELECT id_emp, COUNT(*) as total_actuaciones
        FROM contactos_con_empleados
        GROUP BY id_emp
    """, con)

    plt.figure(figsize=(10, 6))
    plt.bar(df['id_emp'].astype(str), df['total_actuaciones'])
    plt.title('Número Total de Actuaciones por Empleado')
    plt.xlabel('ID del Empleado')
    plt.ylabel('Total de Actuaciones')
    plt.show()

    # Gráfico de actuaciones por día de la semana
    df = pd.read_sql("SELECT fecha FROM contactos_con_empleados", con)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['dia_semana'] = df['fecha'].dt.day_name()
    dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    counts = df['dia_semana'].value_counts().reindex(dias_orden, fill_value=0)

    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.values)
    plt.title('Total de Actuaciones por Día de la Semana')
    plt.xlabel('Día de la Semana')
    plt.ylabel('Total de Actuaciones')
    plt.xticks(rotation=45)
    plt.show()

generar_graficas(con)