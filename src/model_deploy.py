from fastapi import FastAPI, UploadFile,File
from pydantic import BaseModel
from ft_engineering import data_cleaning
from typing import List
import os
import io
import joblib
import uvicorn
import pandas as pd
CP = os.path.dirname(__file__)
MODEL_PATH = os.path.join(CP,'Rf_modelpipeline.pkl')

app= FastAPI()
try:
    model= joblib.load(MODEL_PATH)
except Exception as e:
     print('there is no model available')

    
class LoadPredictionInput(BaseModel):
    capital_prestado: float
    plazo_meses: float
    edad_cliente: float
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje_datacredito: float
    cant_creditosvigentes: float
    huella_consulta: float
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: float
    creditos_sectorCooperativo: float
    creditos_sectorReal: float
    promedio_ingresos_datacredito: float

    # Categóricas nominales
    tipo_laboral: str
    tendencia_ingresos: str

    # Categórica ordinal
    tipo_credito: float

class BatchPredictionInput(BaseModel):
    data: List[LoadPredictionInput]


@app.post("/predict")
def predict(input_data: LoadPredictionInput):   
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)
        return {"prediction": int(prediction[0])}

@app.post("/upload-csv")


async def upload_csv(file: UploadFile = File(...)):
    try:
        contenido = await file.read()
        data = pd.read_csv(io.BytesIO(contenido))
        X,y = data_cleaning(data)
    
        prediction = model.predict(X)
        return {
            "prediction": prediction.tolist()
        }
    except Exception as e:
         print(f'por favor revise su archivo debido al error : {e}')
         
if __name__ == '__main__':
    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=True)