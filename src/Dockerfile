FROM python:3.11-slim

WORKDIR /app

COPY  requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY model_deploy.py .
COPY Rf_modelpipeline.pkl .
COPY ft_engineering.py .
COPY cargar_datos.py .

EXPOSE 5000

CMD ["uvicorn","model_deploy:app", "--host=0.0.0.0", "--port=5000"]