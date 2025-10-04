

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import requests

# --- CONFIGURACIÓN ---
app = Flask(__name__)
CORS(app)

# ¡IMPORTANTE! Pega aquí tu API Key de OpenWeatherMap
OPENWEATHERMAP_API_KEY = "b9887004fb83b6baf80ea22a539cc923"

# --- ORDEN EXACTO DE CARACTERÍSTICAS DEL ENTRENAMIENTO ---
# Esta lista DEBE coincidir con el orden que el modelo aprendió.
FEATURE_ORDER = [
    'T2M_lag_1', 'RH2M_lag_1', 'WS2M_lag_1', 'PS_lag_1',
    'T2M_lag_2', 'RH2M_lag_2', 'WS2M_lag_2', 'PS_lag_2',
    'T2M_lag_3', 'RH2M_lag_3', 'WS2M_lag_3', 'PS_lag_3',
    'T2M_lag_4', 'RH2M_lag_4', 'WS2M_lag_4', 'PS_lag_4',
    'T2M_lag_5', 'RH2M_lag_5', 'WS2M_lag_5', 'PS_lag_5',
    'T2M_lag_6', 'RH2M_lag_6', 'WS2M_lag_6', 'PS_lag_6',
    'month'
]

# --- CARGA DE TODOS LOS MODELOS ---
MODELS = {}
TARGET_VARIABLES = ['T2M', 'RH2M', 'WS2M', 'PS']
print("Cargando modelos de predicción...")
for var in TARGET_VARIABLES:
    try:
        filename = f'model_{var}.joblib'
        MODELS[var] = joblib.load(filename)
        print(f"✅ Modelo '{filename}' cargado.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo '{filename}': {e}")

# --- FUNCIÓN INTELIGENTE PARA PREPARAR DATOS ---
def prepare_features_from_live_weather(lat, lng):
    if not OPENWEATHERMAP_API_KEY or OPENWEATHERMAP_API_KEY == "PEGA_AQUI_TU_API_KEY_DE_OPENWEATHERMAP":
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        response = requests.get(url)
        live_data = response.json()
        
        current_temp = live_data['main']['temp']
        current_humidity = live_data['main']['humidity']
        current_pressure = live_data['main']['pressure'] / 10
        current_wind = live_data['wind']['speed']

        features = {}
        for lag in range(1, 7):
            features[f'T2M_lag_{lag}'] = current_temp + np.random.uniform(-3, 3) - lag * 0.5
            features[f'RH2M_lag_{lag}'] = np.clip(current_humidity + np.random.uniform(-10, 10) - lag, 20, 100)
            features[f'PS_lag_{lag}'] = current_pressure + np.random.uniform(-0.5, 0.5)
            features[f'WS2M_lag_{lag}'] = max(0, current_wind + np.random.uniform(-1, 1))
        
        features['month'] = pd.Timestamp.now().month
        return features
    except Exception as e:
        print(f"Error preparando features: {e}")
        return None

# --- RUTAS DE LA API ---
@app.route('/')
def home():
    return jsonify({
        'message': 'API de clima inteligente para Lima está en línea',
        'models_loaded_successfully': list(MODELS.keys())
    })

@app.route('/predict', methods=['GET'])
def predict():
    if not MODELS:
        return jsonify({'error': 'Ningún modelo disponible.'}), 500

    lat = request.args.get('lat')
    lng = request.args.get('lng')
    if not lat or not lng:
        return jsonify({'error': 'Parámetros "lat" y "lng" son requeridos.'}), 400

    features = prepare_features_from_live_weather(float(lat), float(lng))
    if features is None:
        return jsonify({'error': 'No se pudieron obtener o simular los datos históricos.'}), 500
    
    try:
        input_df_unordered = pd.DataFrame(features, index=[0])
        
        # --- ¡LA CORRECCIÓN MÁGICA ESTÁ AQUÍ! ---
        # Forzamos el DataFrame a tener el orden exacto de columnas que el modelo espera.
        input_df_ordered = input_df_unordered[FEATURE_ORDER]

        predictions = {}
        for var, model in MODELS.items():
            # Usamos el DataFrame ordenado para la predicción
            prediction = model.predict(input_df_ordered)
            predictions[var] = round(float(prediction[0]), 2)
        
        response = {
            "success": True,
            "location": f"Lima, Peru (lat: {lat}, lon: {lng})",
            "prediction_for_next_month": {
                "temperature_celsius": predictions.get('T2M'),
                "humidity_percent": predictions.get('RH2M'),
                "wind_speed_ms": predictions.get('WS2M'),
                "pressure_kpa": predictions.get('PS')
            }
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Ocurrió un error en la predicción del modelo: {str(e)}'}), 500