import sqlite3
import json
from datetime import datetime
import base64
import pandas as pd
import requests
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import graphviz
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
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
import io
app = Flask(__name__)

# --- Variables Globales para Modelos ---
clf_rf = None
clf_logreg = None
clf_dt = None
scaler_logreg = None

# Características y nombres de clases
feature_cols = ['cliente', 'duracion', 'es_mantenimiento', 'satisfaccion_cliente', 'tipo_incidencia']
class_names = ['No Crítico', 'Crítico']



def generar_grafica_arbol_base64(model, feature_names_list, class_names_list, title="Visualización del Árbol"):
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=feature_names_list,
                               class_names=class_names_list,
                               filled=True, rounded=True,
                               special_characters=True, proportion=False, impurity=True)
    try:
        graph = graphviz.Source(dot_data, format="png")
        img_bytes = graph.pipe()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64
    except graphviz.backend.execute.CalledProcessError as e:
        app.logger.error(f"Error al generar gráfica con Graphviz (árbol): {e}")
        app.logger.error("Asegúrate de que Graphviz (el software) esté instalado y en el PATH.")
        return None
    except Exception as e_gen:
        app.logger.error(f"Error general al generar gráfica de árbol: {e_gen}")
        return None


def generar_grafica_coeficientes_base64(model, feature_names_list, title="Importancia de Características"):
    if not hasattr(model, 'coef_'):
        app.logger.error("El modelo no tiene el atributo 'coef_'. No se puede generar gráfica de coeficientes.")
        return None

    coeficientes = model.coef_[0]
    if len(coeficientes) != len(feature_names_list):
        app.logger.error(
            f"Discrepancia en número de coeficientes ({len(coeficientes)}) y nombres de características ({len(feature_names_list)})")
        return None

    sorted_indices = np.argsort(np.abs(coeficientes))[::-1]

    plt.figure(figsize=(10, max(6, len(feature_names_list) * 0.5)))
    plt.barh(np.array(feature_names_list)[sorted_indices], coeficientes[sorted_indices], color="dodgerblue")
    plt.xlabel("Valor del Coeficiente (Impacto en la predicción de 'Crítico')")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_base64


# --- Funciones de Entrenamiento ---
def cargar_y_preparar_datos():
    """Carga datos de data_clasified.json y realiza preprocesamiento, incluyendo 'duracion'."""
    global feature_cols  # Para asegurar que usamos la lista actualizada
    try:
        with open('data_clasified.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        tickets_data = data['tickets_emitidos']

        # Crear una lista de diccionarios para construir el DataFrame fácilmente
        processed_tickets = []
        for ticket in tickets_data:
            duracion = calcular_duracion(ticket['fecha_apertura'], ticket['fecha_cierre'])
            processed_ticket = {
                'cliente': int(ticket['cliente']),
                'duracion': duracion,  # Nueva característica
                'es_mantenimiento': 1 if ticket['es_mantenimiento'] else 0,
                'satisfaccion_cliente': int(ticket['satisfaccion_cliente']),
                'tipo_incidencia': int(ticket['tipo_incidencia']),
                'es_critico': 1 if ticket['es_critico'] else 0
            }
            processed_tickets.append(processed_ticket)

        df = pd.DataFrame(processed_tickets)

        # Verificar que todas las columnas necesarias estén presentes
        # 'es_critico' es el target, feature_cols son las características
        required_cols_for_training = feature_cols + ['es_critico']
        if not all(col in df.columns for col in required_cols_for_training):
            missing_cols = [col for col in required_cols_for_training if col not in df.columns]
            raise KeyError(
                f"Faltan columnas requeridas en el DataFrame procesado: {missing_cols}. Columnas actuales: {df.columns.tolist()}")

        X = df[feature_cols]  # Usar la lista global actualizada
        y = df['es_critico']
        return X, y, df
    except FileNotFoundError:
        app.logger.error("Error: El archivo 'data_clasified.json' no fue encontrado.")
        raise
    except json.JSONDecodeError:
        app.logger.error("Error: El archivo 'data_clasified.json' no es un JSON válido.")
        raise
    except KeyError as e:
        app.logger.error(f"Error de clave al procesar 'data_clasified.json': {e}")
        raise
    except Exception as e_load:
        app.logger.error(f"Error inesperado al cargar y preparar datos: {e_load}", exc_info=True)
        raise


def entrenar_modelo_rf():
    global clf_rf
    print("Entrenando modelo Random Forest...")
    try:
        X, y, _ = cargar_y_preparar_datos()
        clf_rf = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=10)
        clf_rf.fit(X, y)
        print("Modelo Random Forest entrenado exitosamente.")
    except Exception as e:
        print(f"Error al entrenar Random Forest: {e}")
        clf_rf = None

def entrenar_modelo_logreg():
    global clf_logreg, scaler_logreg
    print("Entrenando modelo Regresión Logística...")
    try:
        X, y, _ = cargar_y_preparar_datos()
        scaler_logreg = StandardScaler()
        X_scaled = scaler_logreg.fit_transform(X)
        clf_logreg = LogisticRegression(random_state=0, solver='liblinear')
        clf_logreg.fit(X_scaled, y)
        print("Modelo Regresión Logística entrenado exitosamente.")
    except Exception as e:
        print(f"Error al entrenar Regresión Logística: {e}")
        clf_logreg = None
        scaler_logreg = None

def entrenar_modelo_dt():
    global clf_dt
    print("Entrenando modelo Árbol de Decisión...")
    try:
        X, y, _ = cargar_y_preparar_datos()
        clf_dt = DecisionTreeClassifier(max_depth=5, random_state=0)
        clf_dt.fit(X, y)
        print("Modelo Árbol de Decisión entrenado exitosamente.")
    except Exception as e:
        print(f"Error al entrenar Árbol de Decisión: {e}")
        clf_dt = None


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


# Función para calcular la duración del ticket en días
def calcular_duracion(fecha_apertura, fecha_cierre):
    apertura = datetime.strptime(fecha_apertura, "%Y-%m-%d")
    cierre = datetime.strptime(fecha_cierre, "%Y-%m-%d")
    return (cierre - apertura).days

    # Función para calcular el tiempo total de contacto con empleados
def calcular_tiempo_contacto(contactos):
    return sum(contacto['tiempo'] for contacto in contactos)


@app.route('/predict_critical', methods=['GET', 'POST'])
def predict_critical_ticket():
    global clf_rf, clf_logreg, clf_dt, scaler_logreg, feature_cols, class_names

    prediction_text = None
    error_message = None
    model_error = None
    form_data = request.form

    tree_images_data_rf = []
    single_model_image_data = None

    # Re-entrenar si algún modelo es None (salvaguarda)
    if clf_rf is None: entrenar_modelo_rf()
    if clf_logreg is None or scaler_logreg is None: entrenar_modelo_logreg()  # Asegurar que scaler también se entrena
    if clf_dt is None: entrenar_modelo_dt()

    if request.method == 'POST':
        try:
            model_selection = request.form.get('model_selection')
            if not model_selection:
                raise ValueError("No se seleccionó ningún modelo.")

            # Obtener datos del formulario
            cliente_id_str = request.form.get('cliente_id')
            fecha_apertura_str = request.form.get('fecha_apertura')  # NUEVO
            fecha_cierre_str = request.form.get('fecha_cierre')  # NUEVO
            es_mantenimiento_str = request.form.get('es_mantenimiento')
            satisfaccion_str = request.form.get('satisfaccion')
            tipo_incidencia_str = request.form.get('tipo_incidencia')

            # Validar y convertir datos
            if not all([cliente_id_str, fecha_apertura_str, fecha_cierre_str, satisfaccion_str, tipo_incidencia_str]):
                raise ValueError("Todos los campos son requeridos (excepto ¿Es Mantenimiento?).")

            cliente_id = int(cliente_id_str)
            # fecha_apertura y fecha_cierre se usan para calcular duración
            es_mantenimiento = 1 if es_mantenimiento_str == '1' else 0
            satisfaccion = int(satisfaccion_str)
            tipo_incidencia = int(tipo_incidencia_str)

            if not (0 <= satisfaccion <= 10):
                raise ValueError("Satisfacción debe estar entre 0 y 10.")

            # Calcular duración para la predicción
            duracion_pred = calcular_duracion(fecha_apertura_str, fecha_cierre_str)
            if duracion_pred < 0:  # Validación adicional
                raise ValueError("La fecha de cierre no puede ser anterior a la fecha de apertura.")

            # Preparar DataFrame de entrada para la predicción
            # Asegurarse de que el orden coincida con `feature_cols`
            input_data_dict = {
                'cliente': [cliente_id],
                'duracion': [duracion_pred],  # Añadida duración
                'es_mantenimiento': [es_mantenimiento],
                'satisfaccion_cliente': [satisfaccion],
                'tipo_incidencia': [tipo_incidencia]
            }
            input_features = pd.DataFrame(input_data_dict)[feature_cols]  # Reordenar según feature_cols

            current_model = None
            model_display_name = ""
            prediction = None
            proba = None

            if model_selection == 'random_forest':
                if clf_rf is None:
                    model_error = "El modelo Random Forest no está disponible o no pudo ser entrenado."
                    raise ValueError(model_error)
                current_model = clf_rf
                model_display_name = "Random Forest"
                prediction = current_model.predict(input_features)
                proba = current_model.predict_proba(input_features)
                for i, estimator in enumerate(current_model.estimators_[:min(3,
                                                                             len(current_model.estimators_))]):  # Limitar a 3 para no sobrecargar
                    img_data = generar_grafica_arbol_base64(estimator, feature_cols, class_names,
                                                            title=f"Árbol {i + 1} de {model_display_name}")
                    if img_data: tree_images_data_rf.append(img_data)

            elif model_selection == 'linear_regression':  # Asumimos Logística
                if clf_logreg is None or scaler_logreg is None:
                    model_error = "El modelo de Regresión Logística no está disponible o no pudo ser entrenado."
                    raise ValueError(model_error)
                current_model = clf_logreg
                model_display_name = "Regresión Logística"
                input_features_scaled = scaler_logreg.transform(input_features)
                prediction = current_model.predict(input_features_scaled)
                proba = current_model.predict_proba(input_features_scaled)
                single_model_image_data = generar_grafica_coeficientes_base64(current_model, feature_cols,
                                                                              title=f"Coeficientes - {model_display_name}")

            elif model_selection == 'decision_tree':
                if clf_dt is None:
                    model_error = "El modelo de Árbol de Decisión no está disponible o no pudo ser entrenado."
                    raise ValueError(model_error)
                current_model = clf_dt
                model_display_name = "Árbol de Decisión"
                prediction = current_model.predict(input_features)
                proba = current_model.predict_proba(input_features)
                single_model_image_data = generar_grafica_arbol_base64(current_model, feature_cols, class_names,
                                                                       title=f"Visualización - {model_display_name}")

            else:
                raise ValueError("Modelo de predicción no válido seleccionado.")

            if prediction is not None and proba is not None:
                resultado_clase = class_names[prediction[0]]
                prediction_text = (
                    f"El ticket analizado con <strong>{model_display_name}</strong> es: <strong>{resultado_clase}</strong><br>"
                    f"(Probabilidades: {class_names[0]}={proba[0][0]:.2f}, "
                    f"{class_names[1]}={proba[0][1]:.2f})")
            else:
                error_message = "No se pudo realizar la predicción."


        except ValueError as ve:
            error_message = f"Error en los datos introducidos o selección: {ve}"
        except KeyError as ke:  # Aunque la validación anterior debería cubrir esto
            error_message = f"Falta un campo en el formulario: {ke}"
        except Exception as e:
            error_message = f"Error inesperado durante la predicción: {e}"
            app.logger.error(
                f"Error en predicción ({model_selection if 'model_selection' in locals() else 'desconocido'}): {e}",
                exc_info=True)

    return render_template('predict_critical.html',
                           prediction_text=prediction_text,
                           error=error_message,
                           model_error=model_error,
                           form_data=form_data,
                           tree_images_data=tree_images_data_rf,
                           regression_image_data=single_model_image_data)


# --- Llamadas de Entrenamiento al Inicio ---
try:
    entrenar_modelo_rf()
    entrenar_modelo_logreg()
    entrenar_modelo_dt()
except Exception as e_startup:
    print(f"Error crítico durante el entrenamiento inicial de modelos: {e_startup}")
    # Considerar cómo manejar esto (¿detener la app?)

if __name__ == '__main__':
    app.run(debug=False)
