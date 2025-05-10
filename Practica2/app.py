from datetime import datetime
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, send_file, request
from fpdf import FPDF
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
from reportlab.lib import colors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sqlite3
import json
import base64
import pandas as pd
import requests
import graphviz
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os
import io
matplotlib.use('Agg')
app = Flask(__name__)

# --- Variables Globales para Modelos ---
clf_rf = None
clf_linreg = None
clf_dt = None
scaler_linreg = None

feature_cols = ['cliente', 'duracion', 'es_mantenimiento', 'satisfaccion_cliente', 'tipo_incidencia']
class_names = ['No Crítico', 'Crítico']
validation_metrics = {}

# --- Variables Globales para Conjuntos de Datos Divididos ---
X_train_global, X_val_global, X_test_global = None, None, None
y_train_global, y_val_global, y_test_global = None, None, None

# --- Abrir Datos desde JSON ---
with open('data_clasified.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
tickets_data = data['tickets_emitidos']

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
        app.logger.error("Modelo sin 'coef_'.")
        return None

    coeficientes = model.coef_
    if len(coeficientes) != len(feature_names_list):
        app.logger.error("Discrepancia coefs/features.")
        return None

    fig = Figure(figsize=(10, max(6, len(feature_names_list) * 0.5)))
    ax = fig.subplots()
    sorted_indices = np.argsort(np.abs(coeficientes))[::-1]
    ax.barh(np.array(feature_names_list)[sorted_indices], coeficientes[sorted_indices], color="dodgerblue")
    ax.set_xlabel("Valor del Coeficiente")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    return base64.b64encode(img_buffer.getvalue()).decode('utf-8')


# --- Carga y Preparación de Datos ---
def cargar_y_preparar_datos_globales():
    global X_train_global, X_val_global, X_test_global, y_train_global, y_val_global, y_test_global, feature_cols
    app.logger.info("Cargando y preparando datos globales...")
    try:
        processed_tickets = []
        for ticket in tickets_data:
            fecha_apertura = ticket.get('fecha_apertura')
            fecha_cierre = ticket.get('fecha_cierre')
            duracion = calcular_duracion(fecha_apertura, fecha_cierre) if fecha_apertura and fecha_cierre else 0
            processed_ticket = {
                'cliente': int(ticket.get('cliente', 0)), 'duracion': duracion,
                'es_mantenimiento': 1 if ticket.get('es_mantenimiento') else 0,
                'satisfaccion_cliente': int(ticket.get('satisfaccion_cliente', 0)),
                'tipo_incidencia': int(ticket.get('tipo_incidencia', 0)),
                'es_critico': 1 if ticket.get('es_critico') else 0
            }
            processed_tickets.append(processed_ticket)

        df = pd.DataFrame(processed_tickets)

        required_cols_for_training = feature_cols + ['es_critico']
        if not all(col in df.columns for col in required_cols_for_training):
            missing_cols = [col for col in required_cols_for_training if col not in df.columns]
            raise KeyError(f"Faltan columnas: {missing_cols}. Actuales: {df.columns.tolist()}")

        X = df[feature_cols]
        y = df['es_critico']


        X_train_temp, X_test_global, y_train_temp, y_test_global = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y if y.nunique() > 1 and y.value_counts().min() >= 2 else None)

        X_train_global, X_val_global, y_train_global, y_val_global = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42, stratify=y_train_temp)

        app.logger.info(f"Datos cargados y divididos: Ent={len(X_train_global) if X_train_global is not None else 0}, Val={len(X_val_global) if X_val_global is not None else 0}, Test={len(X_test_global) if X_test_global is not None else 0}")

    except Exception as e_load:
        app.logger.error(f"Error fatal al preparar datos globales: {e_load}", exc_info=True)
        # Dejar las variables globales como None para que los modelos no se entrenen
        X_train_global, X_val_global, X_test_global, y_train_global, y_val_global, y_test_global = [None] * 6


# --- Funciones de Entrenamiento de Modelos ---
def entrenar_modelo_rf(X_train, y_train):
    global clf_rf
    if X_train is None or y_train is None or X_train.empty:
        app.logger.error("Random Forest: Datos de entrenamiento no disponibles. No se puede entrenar.")
        clf_rf = None
        return
    try:
        clf_rf = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=10)
        clf_rf.fit(X_train, y_train)
        app.logger.info("Modelo Random Forest entrenado exitosamente.")
    except Exception as e:
        app.logger.error(f"Error al entrenar Random Forest: {e}", exc_info=True)
        clf_rf = None


def entrenar_modelo_linreg(X_train, y_train):
    global clf_linreg
    if X_train is None or y_train is None or X_train.empty:
        app.logger.error(f"Regresión Lineal: Datos de entrenamiento no disponibles.")
        clf_linreg = None
        return
    try:
        clf_linreg = LinearRegression()
        clf_linreg.fit(X_train, y_train)
        app.logger.info(f"Modelo Regresión Lineal entrenado exitosamente.")

    except Exception as e:
        app.logger.error(f"Error al entrenar Regresión Lineal: {e}", exc_info=True);
        clf_linreg, scaler_linreg = None, None


def entrenar_modelo_dt(X_train, y_train):
    global clf_dt
    app.logger.info("Entrenando modelo Árbol de Decisión...")
    if X_train is None or y_train is None or X_train.empty:
        app.logger.error("Árbol de Decisión: Datos de entrenamiento no disponibles.")
        clf_dt = None
        return
    try:
        clf_dt = DecisionTreeClassifier(max_depth=5, random_state=0)
        clf_dt.fit(X_train, y_train)
        app.logger.info("Modelo Árbol de Decisión entrenado exitosamente.")
    except Exception as e:
        app.logger.error(f"Error al entrenar Árbol de Decisión: {e}", exc_info=True)
        clf_dt = None



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
@app.route('/top_clientes')
def get_top_clientes():
    x = request.args.get('num_clientes')
    if x is not None:
        x = int(x)
        if x <= 0: x = 5
    else: x = 5
    #Ordenar y escoger los X clientes requeridos
    cliMaxIncidents_ordenado = calculateTopClientes(x)

    tabla_html = cliMaxIncidents_ordenado.to_html(index=False, classes='data')
    return render_template('top_clientes.html', tabla_clientes_html=tabla_html, current_num_clientes=x)

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

@app.route('/top_tipos', methods=['GET'])
def get_top_tipos():
    num_tipos = request.args.get('num_clientes')
    if num_tipos is not None:
        num_tipos = int(num_tipos)
        if num_tipos <= 0: num_tipos = 5
    else:
        num_tipos = 5

    top_tipos_data = calculateTopTipos(num_tipos)
    if top_tipos_data.empty:
        tabla_html = "<p>No hay datos de tipos de incidencia para mostrar.</p>"
    else:
        tabla_html = top_tipos_data.to_html(index=False, classes='data')

        # Pasar el número actual a la plantilla para el PDF y el valor del input
    return render_template('top_tipos.html', tabla_tipo_html=tabla_html, current_num_tipos=num_tipos)

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

@app.route('/top_tipos/downloadPDF', methods=['GET'])
def generateTopTiposPDF():
    try:
        num_tipos_str = request.args.get('num_tipos', '5')
        x = int(num_tipos_str)
        if x <= 0:
            x = 5
    except ValueError:
        x = 5

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

@app.route('/predict_critical', methods=['GET','POST'])
def predict_critical_ticket():
    global clf_rf, clf_logreg, clf_dt, scaler_logreg, feature_cols, class_names, validation_metrics

    prediction_text = None
    error_message = None
    model_error = None
    form_data = request.form
    tree_images_data_rf = []
    single_model_image_data = None
    confusion_matrix_image = None

    if request.method == 'POST':
        try:
            model_selection = request.form.get('model_selection')
            if not model_selection: raise ValueError("No se seleccionó ningún modelo.")

            cliente_id_str = request.form.get('cliente_id')
            fecha_apertura_str = request.form.get('fecha_apertura')
            fecha_cierre_str = request.form.get('fecha_cierre')
            es_mantenimiento_str = request.form.get('es_mantenimiento')
            satisfaccion_str = request.form.get('satisfaccion')
            tipo_incidencia_str = request.form.get('tipo_incidencia')

            if not all([cliente_id_str, fecha_apertura_str, fecha_cierre_str, satisfaccion_str, tipo_incidencia_str]):
                raise ValueError("Todos los campos (excepto ¿Es Mantenimiento?) son requeridos.")

            cliente_id = int(cliente_id_str)
            es_mantenimiento = 1 if es_mantenimiento_str == '1' else 0
            satisfaccion = int(satisfaccion_str)
            tipo_incidencia = int(tipo_incidencia_str)

            if not (0 <= satisfaccion <= 10): raise ValueError("Satisfacción entre 0 y 10.")

            duracion_pred = calcular_duracion(fecha_apertura_str, fecha_cierre_str)
            if duracion_pred < 0: raise ValueError("Fecha de cierre antes de apertura.")

            input_data_dict = {
                'cliente': [cliente_id], 'duracion': [duracion_pred],
                'es_mantenimiento': [es_mantenimiento],
                'satisfaccion_cliente': [satisfaccion], 'tipo_incidencia': [tipo_incidencia]
            }
            input_features = pd.DataFrame(input_data_dict)[feature_cols]

            if model_selection == 'random_forest':
                if clf_rf is None:
                    model_error = "Random Forest no disponible."
                    raise ValueError(model_error)
                current_model, model_display_name = clf_rf, "Random Forest"
                prediction = current_model.predict(input_features)
                proba = current_model.predict_proba(input_features)
                for i, estimator in enumerate(current_model.estimators_[:min(3, len(current_model.estimators_))]):
                    img_data = generar_grafica_arbol_base64(estimator, feature_cols, class_names, title=f"Árbol {i + 1} RF")
                    if img_data: tree_images_data_rf.append(img_data)

                if prediction is not None and proba is not None:
                    resultado_clase = class_names[prediction[0]]
                    prediction_text = (f"Ticket ({model_display_name}): <strong>{resultado_clase}</strong><br>"
                                       f"(Probabilidades: {class_names[0]}={proba[0][0]:.2f}, {class_names[1]}={proba[0][1]:.2f})")
                else: error_message = "No se pudo predecir."

            elif model_selection == 'linear_regression':
                if clf_linreg is None:
                    model_error = "Reg. Lineal no disponible."
                    raise ValueError(model_error)
                current_model, model_display_name = clf_linreg, "Regresión Lineal"

                prediction = current_model.predict(input_features)
                raw_score = current_model.predict(input_features)[0]  # Obtener puntuación cruda

                single_model_image_data = generar_grafica_coeficientes_base64(current_model, feature_cols, title=f"Coeficientes - {model_display_name}")

                if prediction is not None:
                    # Aplicar umbral para decidir la clase
                    threshold = 0.5
                    resultado_clase = 1 if raw_score >= threshold else 0
                    prediction_text = (
                        f"Ticket ({model_display_name}): <strong>{class_names[resultado_clase]}</strong><br>"
                        f"(Puntuación cruda: {raw_score:.3f}, Umbral: {threshold})")
                else: error_message = "No se pudo predecir."

            elif model_selection == 'decision_tree':
                if clf_dt is None:
                    model_error = "Árbol Decisión no disponible."
                    raise ValueError(model_error)
                current_model, model_display_name = clf_dt, "Árbol de Decisión"
                prediction = current_model.predict(input_features)
                proba = current_model.predict_proba(input_features)
                single_model_image_data = generar_grafica_arbol_base64(current_model, feature_cols, class_names, title=f"Vis - {model_display_name}")

                if prediction is not None and proba is not None:
                    resultado_clase = class_names[prediction[0]]
                    prediction_text = (f"Ticket ({model_display_name}): <strong>{resultado_clase}</strong><br>"
                                       f"(Probabilidades: {class_names[0]}={proba[0][0]:.2f}, {class_names[1]}={proba[0][1]:.2f})")
                else: error_message = "No se pudo predecir."
                    
            else:
                raise ValueError("Modelo no válido.")

        except ValueError as ve:
            error_message = f"Error datos/selección: {ve}"
        except KeyError as ke:
            error_message = f"Falta campo: {ke}"
        except Exception as e:
            error_message = f"Error inesperado: {e}"
            app.logger.error(f"Error predicción: {e}", exc_info=True)

    return render_template('predict_critical.html',
                           prediction_text=prediction_text, error=error_message, model_error=model_error,
                           form_data=form_data, tree_images_data=tree_images_data_rf,
                           regression_image_data=single_model_image_data,
                           confusion_matrix_image=confusion_matrix_image)


# --- Carga inicial de datos y entrenamiento de modelos ---
try:
    cargar_y_preparar_datos_globales()
    entrenar_modelo_rf(X_train_global, y_train_global)
    entrenar_modelo_linreg(X_train_global, y_train_global)
    entrenar_modelo_dt(X_train_global, y_train_global)
except Exception as e_startup:
    app.logger.error(f"Error crítico durante la inicialización y entrenamiento: {e_startup}", exc_info=True)


if __name__ == '__main__':
    app.run(debug=False)
