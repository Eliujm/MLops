import pandas as pd
import numpy as np
import joblib
import cargar_datos as cd
import os
import sys
from ft_engineering import processing_pipeline,data_cleaning
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,confusion_matrix,classification_report,make_scorer,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC

CD = os.path.dirname(__file__)
DATA_DR = os.path.dirname(CD)
DATA_PATH = os.path.join(DATA_DR,'base_de_datos.xlsx')
def summarize_classification(y, y_hat):
    """
    métricas básicas de clasificación
    """

    return {
        "accuracy": accuracy_score(y, y_hat),
        "precision": precision_score(y, y_hat, zero_division=0),
        "recall": recall_score(y, y_hat, zero_division=0),
        "f1": f1_score(y, y_hat, zero_division=0),
    }

def build_model(path = DATA_PATH):
    df = cd.cargar_datos(path)

    #spliting the data into features and target#

    X,y = data_cleaning(df)
    #classifiying the featuress#
   

    num_features =['capital_prestado','plazo_meses','edad_cliente','total_otros_prestamos','cuota_pactada','puntaje_datacredito','huella_consulta','saldo_mora','saldo_total','saldo_principal','creditos_sectorFinanciero','creditos_sectorCooperativo','creditos_sectorReal','promedio_ingresos_datacredito']

    nom_features = ['tipo_laboral']

    or_features = ['tendencia_ingresos']

    ti_order = ['Decreciente','Estable','Creciente']

    # spliting into training and test#
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.2, random_state= True, stratify= y)
    # getting the preprocessor ready with the features #
    preprocessor = processing_pipeline(num_features,nom_features,or_features,ti_order)


   

    Rf_model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                        criterion='gini', max_depth=None, max_features='sqrt',
                        max_leaf_nodes=None, max_samples=None,
                        min_impurity_decrease=0.0, min_samples_leaf=1,
                        min_samples_split=2, min_weight_fraction_leaf=0.0,
                        monotonic_cst=None, n_estimators=100, n_jobs=-1,
                        oob_score=False, random_state=42, verbose=0,
                        warm_start=False)
    # entrenando el modelo individual# 

    Model_pipeline = Pipeline([(
        'preprocesor',preprocessor),
        ('model',Rf_model)]

    )
    Model_pipeline.fit(x_train,y_train)
    #  obteniendo las principales del modelo #
    y_hat_proba = Model_pipeline.predict_proba(x_test)[:,1]
    y_hat = Model_pipeline.predict(x_test)


    Class_Report = classification_report(y_test, y_hat,zero_division= 0)
    Conf_Matrix = confusion_matrix(y_test, y_hat)
    Auc_Report = roc_auc_score(y_test,y_hat_proba)

    # creando el pkl para ser usado desplegado más adelandte #
    joblib.dump(Model_pipeline, 'Rf_modelpipeline.pkl')

  

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        
        "precision_0": make_scorer(precision_score, pos_label=0,zero_division =0),
        "precision_1": make_scorer(precision_score, pos_label=1,zero_division =0),
        
        "recall_0": make_scorer(recall_score, pos_label=0),
        "recall_1": make_scorer(recall_score, pos_label=1),
        
        "f1_0": make_scorer(f1_score, pos_label=0),
        "f1_1": make_scorer(f1_score, pos_label=1)
    }

    cv_scores = cross_validate(
        Model_pipeline,
        x_train,
        y_train,
        cv=5,
        scoring=scoring
    )

    print("nombre del modelo: RandomForestClassifier")
    print("-"*30)

    for metric in scoring.keys():
        print(f"{metric}_avg: {cv_scores['test_'+metric].mean()}")
        print(f"{metric}_std: {cv_scores['test_'+metric].std()}")
        print("-"*30)


    return Class_Report,Conf_Matrix, Auc_Report 
    
if __name__ == "__main__":
  
    if os.path.exists(DATA_PATH):
        build_model()
    else:
        print("File does not exist")
    