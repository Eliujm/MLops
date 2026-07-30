Prediccion de comportamiento de cliente
-------------------------------------
-------------------------------------


La empresa de finanaciamiente espera que con datos historicos podamos contruir, entrenar y poner un modelo en producción que permita predecier el comportamiento del cliente, se recibieron los datos con las siguientes condiciones:

*95.3% de los clientes realizaran pagos a tiempo
*4.7% es el porcentaje de clientes que tendrá retraso.

se observó la distribucón según las categorías
---------------------------------------------

*cerca de un 63% del espacio de clientes esta compuesto por empleados y el restante 37% restante de independientes.
* los empleados tiene la mayor proporcien en la categoría de ingresos crecientes y también en la decreciente.

Modelo y resultados
-------------------

Se entrenaron 4 modelos pricipales :GradientBoostingClassifier,  RidgeClassifier,LinearDiscriminantAnalysis,RandomForestClassifier, se comparó las metricas AUC, learning curve y variación en el accuracy, el mejor modelo basado en esas metrias fue el RandomForestClassifier, el mejor performane es obtenido con un holdout de 30% o 70/30.

Estos son los dato basados en la clase 1 para el caso de recall y precision, se llevó acabo con una validación cruzada.
    RandomForesClassifier:
    AUC :0.6532
    accuracy_avg : 0.8739750000000001
    accuracy_std: 0.2747313634178932,
    precision_avg: 0.874375,
    precision_std: 0.2749189173854059,
    recall_avg: 0.916225,
    recall_std: 0.2882539793149595,
    f1_score_avg: 0.894775,
    f1_score_std: 0.281530276508423


Estos son los reportes del entrenamiento de las 2 clases con una estratificaciín 80/20:
nombre del modelo: RandomForestClassifier
------------------------------
accuracy_avg: 0.9534262485481998
accuracy_std: 0.0006772301852317709
------------------------------
roc_auc_avg: 0.6515459546769294
roc_auc_std: 0.03129670210085261
------------------------------
precision_0_avg: 0.6433333333333333
precision_0_std: 0.3398692559074903
------------------------------
precision_1_avg: 0.953700052068528
precision_1_std: 0.0007475142441576685
------------------------------
recall_0_avg: 0.02688949111713339
recall_0_std: 0.016170024620636027
------------------------------
recall_1_avg: 0.9996341463414634
recall_1_std: 0.000298718261315022
------------------------------
f1_0_avg: 0.051394611892544
f1_0_std: 0.030506011493991014
------------------------------
f1_1_avg: 0.9761267812451944
f1_1_std: 0.0003403980005494833

Estos son  los datos  del set de test con 80/20 
------------------------------------------------
 precision    recall  f1-score   support

           0       0.00      0.00      0.00       102
           1       0.95      1.00      0.98      2051

    accuracy                           0.95      2153
   macro avg       0.48      0.50      0.49      2153
weighted avg       0.91      0.95      0.93      2153

Anque los script estan en proporciones 80/20, debido a desbalance se probó 70/30 y este fue el desempeño
--------------------------------------------------------------------------------------------------------
    precision    recall  f1-score   support

           0       0.80      0.03      0.05       153
           1       0.95      1.00      0.98      3076

    accuracy                           0.95      3229
   macro avg       0.88      0.51      0.51      3229
weighted avg       0.95      0.95      0.93      3229

Conslusión el mejor predictor ante estas condiciones seria el RandomForesClassifier con una proporcion 70/30 para train y test respectivamente, los resultados son notables.

Organizacion del projecto

README.md                    <-  el documento para desarrolladores de como usar el projecto
base_de_datosxlxs            <-  base de datos de entrenamiento
base_de_datos_drift.csv      <- base de datos de prueba drift
__init__.py                  <- archvo para importar los dir como modulos
comprensión_eda.ipynb        <- Analisis exploratorio de datos, analisis unirvariable bivariable, correlación
carga_datos.py               <- script de python para cargar los datos |
                                                                       |__cargar_datos

ft_engineering.py            <-  scrip con la ingenieria de caracteristicas |
                                                                        |__data_cleaning <- remueve impurezas de los datos, estandaraiza columnas ,inconsistencias
                                                                        |
                                                                        |__processiong_pipeline <- imputación, escalado y codificación

model_training_evaluatio.ipynb <- evaluación de los modelos de machine learning principales con pycaret, solo 4 modelos preseleccionas bajo accuracy, recall,  learning curve
model_training.py              <- entrenamiento del best_model escogido RandomForesClassifier y el empacado del modelo mdeiante joblib|
                                                                                                                                      |__summarize_classification
                                                                                                                                      |
                                                                                                                                      |__build_model

model_monitoring.py            <- script en streamlit para desplegarla aplicación de monitoreo  con metricas de drift
drift.report.html               <- coumento generado por monitor monitoring para cargar las metricas drift
prediction_logs.csv             <- archivo csv que contiene los logs de las predicciones de la app de streamlit

model_deploy.py                <- despliegue del modelo en fastAPI  con carga por cliente o batch
Dockerfile                     <- archivo de docker par ala creación de imagen  y correr el script model_deploy.py en un contenedor de docker
requirements.txt               <- documento con toda la información necesaria de depencias para instalar en docker 
.gitignore                      <- archvo con la inf de los archivos que git debe ignorar




