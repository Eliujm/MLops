import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import cargar_datos as cd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import  SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder

def data_cleaning(df:pd.DataFrame):

    if 'Pago_atiempo' in df.columns:
        X = df.drop(['Pago_atiempo','fecha_prestamo', 'tipo_credito','puntaje','cant_creditosvigentes','salario_cliente','saldo_mora_codeudor'] , axis = 1)
        y = df['Pago_atiempo']

    else:
        X = df.drop(['fecha_prestamo', 'tipo_credito','puntaje','cant_creditosvigentes','salario_cliente','saldo_mora_codeudor'] , axis = 1)
        #classifiying the featuress#
    
    
    num_features =['capital_prestado','plazo_meses','edad_cliente','total_otros_prestamos','cuota_pactada','puntaje_datacredito','huella_consulta','saldo_mora','saldo_total','saldo_principal','creditos_sectorFinanciero','creditos_sectorCooperativo','creditos_sectorReal','promedio_ingresos_datacredito']
    ti_order = ['Decreciente','Estable','Creciente']
    for i in num_features:
        X[i] = X[i].apply(lambda x : x if isinstance(x,(int,float) )else np.nan)

    X['tipo_laboral'] = X['tipo_laboral'].apply(lambda x : x if isinstance(x,str)  else np.nan)
    

    X['tendencia_ingresos'] = X['tendencia_ingresos'].apply(lambda x : x if x in ti_order  else np.nan)
    
    if 'Pago_atiempo' in df.columns:
        return X,y
    else:
        return X


def processing_pipeline(num_features:list,nom_features:list,or_features:list, ti_order:list)->object:



    # creating pipe lines to tranfor the data #
    num_transformer = Pipeline(steps=
    [('imputer', SimpleImputer(strategy= 'mean')),
    # optional: ('scaler', StandardScaler()),
    ])
    nom_transformer = Pipeline(steps= [
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('onehot',OneHotEncoder(handle_unknown= 'ignore', sparse_output= False))
    ])
    or_transformer= Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('ordcoder',OrdinalEncoder(categories=[ti_order]))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num',num_transformer,num_features),
        ('nom',nom_transformer,nom_features), 
        ('or',or_transformer,or_features)])
    

    return preprocessor
    


