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
    


