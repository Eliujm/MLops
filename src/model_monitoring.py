import pandas  as pd
import numpy as np
import streamlit as st
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from ft_engineering import data_cleaning
from cargar_datos import cargar_datos
from evidently import Report
from evidently.presets import DataDriftPreset

#rastreando los archivos a cargar#

CUR_FP = os.path.dirname(__file__)
CUR_WD = os.path.dirname(CUR_FP)
MODEL_PATH = os.path.join(CUR_WD,'base_de_datos.xlsx')
MODEL_DRIGT_PATH = os.path.join(CUR_WD,'Base_de_datos_drift.csv')
DRIFT_REPORT_PATH =os.path.join(CUR_WD,'drif_report.html')
LOGS_REPORT_PATH =os.path.join(CUR_FP,'predictions_logs.csv')

logs = None

# declarando las variables de sesion para que no cargen nuevamente cuando se corra el archivo
if 'reference_Data' not in st.session_state:
    st.session_state.reference_Data = cargar_datos(MODEL_PATH)
if 'new_data' not in st.session_state:    
    st.session_state.new_data = cargar_datos(MODEL_DRIGT_PATH)

cols_todrop = ['Pago_atiempo','fecha_prestamo', 'tipo_credito','puntaje','cant_creditosvigentes','salario_cliente','saldo_mora_codeudor']

reference_Data= st.session_state.reference_Data
reference_Data_clean,y_ref= data_cleaning(reference_Data)


st.set_page_config('model drifting evaluation',layout = 'wide')

# cargando el modelo predictor #

@st.cache_resource
def get_model():
    pipeline_model = joblib.load('Rf_modelpipeline.pkl')
    return pipeline_model


st.title('Model drifting evaluation')

st.sidebar.header('Carga de datos', width= 'stretch')

# creado el sistema de archivos temporal #

uploaded_file = st.sidebar.file_uploader('Por favor carge su archivo de datos',
                                 type= ['csv','xlsx'])


def get_predictions(X_data: pd.DataFrame):
    model = get_model()
    
    y_hat_c1 = model.predict_proba(X_data)[:,1]
    y_hat_c0= model.predict_proba(X_data)[:,0]
    return y_hat_c0,y_hat_c1

# leyendo los datos #
data= st.text_input('Por favor introduzca los datos del cliente manualmente en json format')
    
         
if st.button('Hacer predicción'):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else: 
             st.error("Este formato no es soportado")

        # limpiando los datos que fueron cargaados en la app#

       
    elif data:
        if isinstance(data, str):
            data = json.loads(data)
        df = pd.DataFrame([data])
    else:
        st.warning("No hay archivo ni datos JSON")

    st.session_state.X,st.session_state.y= data_cleaning(df)
    st.session_state.outcome_0,st.session_state.outcome_1= get_predictions(st.session_state.X)
    st.session_state.log_data =st.session_state.X.copy()
    st.session_state.log_data['Prediction'] = (st.session_state.outcome_1 >= 0.5).astype(int)
    st.session_state.log_data['timestamp'] = pd.Timestamp.now()
    if os.path.exists(LOGS_REPORT_PATH):
        st.session_state.log_data.to_csv('predictions_logs.csv',mode= 'a',header=False ,index= False)
    else:
        st.session_state.log_data.to_csv('predictions_logs.csv', index= False)

    #analisis de data drift
    st.session_state.report= Report(metrics =[DataDriftPreset()])
    result = st.session_state.report.run(reference_data = reference_Data,
    current_data = st.session_state.X
            )
    result.save_html(DRIFT_REPORT_PATH)
    
    logs = pd.read_csv(LOGS_REPORT_PATH)
    st.session_state.logs = logs
# dividiendo el screen pricipal en tabs #
tab1,tab2,tab3 = st.tabs(['Graficas','Drift','Logs'])
with tab1:
# graficando precicciones
    if logs is not None:
        fg1,ax = plt.subplots(1,2,figsize = (8,6), dpi = 150)
        
        sns.histplot(logs['Prediction'], ax= ax[0])
        ax[0].set_xlabel('Prediciones')
        if 'timestamp' in logs.columns:
                    logs['timestamp'] = pd.to_datetime(logs['timestamp'])
                    # Agrupar por minuto para mejor visualización
                    temporal_data = logs.groupby(
                        logs['timestamp'].dt.floor('min'))['Prediction'].mean().reset_index()

        sns.lineplot(temporal_data, x ='timestamp' ,y= 'Prediction', ax = ax[1])
        
        ax[1].tick_params(axis="x", labelrotation=90)   
        plt.tight_layout()
        st.pyplot(fg1)
    else: 
        st.warning('no data avaliable yet')
       
with tab2:
    if 'report' in st.session_state:
        with open(DRIFT_REPORT_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=1000, scrolling=True)
    else:
            st.warning('no se ha cargado un dataframe')
with tab3:
    st.write("Exists:", os.path.exists(LOGS_REPORT_PATH))

    if  st.button ('Cargar logs'):
        st.dataframe(st.session_state.logs)

# asignando las metricas a la barra lateral #
if 'X' in  st.session_state :
    st.sidebar.metric("Samples", len(st.session_state.X))
    st.sidebar.metric("Average class 1", f"{st.session_state.outcome_1.mean():.2f}")
    st.sidebar.metric("Average class 0", f"{st.session_state.outcome_0.mean():.2f}")
    st.sidebar.metric("Std. deviation class 1", f"{st.session_state.outcome_1.std():.2f}")
    st.sidebar.metric("Std. deviation class 0", f"{st.session_state.outcome_0.std():.2f}")
        

else:
    st.sidebar.metric('Samples :', 0) 
    st.sidebar.metric('Average  class 1:', 0)
    st.sidebar.metric('Average  class 0:', 0)
    st.sidebar.metric('Sd deviation  class 1:', 0)
    st.sidebar.metric('Sd deviation  class 0:', 0)

   





