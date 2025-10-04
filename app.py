# app.py - Tu API de Clima Inteligente para Lima

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import requests # Para llamar a APIs externas
import os

# --- CONFIGURACIÓN ---
app = Flask(__name__)
CORS(app) # Permite que tu app web llame a esta API

# ¡¡MUY IMPORTANTE!!
# 1. Regístrate gratis en https://openweathermap.org/
# 2. Ve a la sección "API keys" en tu perfil y pega tu clave aquí.
OPENWEATHERMAP_API_KEY = "b9887004fb83b6baf80ea22a539cc923"

# --- CARGA DE TODOS TUS MODELOS ENTRENADOS ---
MODELS = {}
# Lista de los modelos que entrenaste
TARGET_VARIABLES = ['T2M', 'RH2M', 'WS2M', 'PS'] 

print("Cargando modelos de predicción...")
for var in TARGET_VARIABLES:
    try:
        # El nombre del archivo debe coincidir exactamente con el que guardaste
        filename = f'model_{var}.joblib'
        MODELS[var] = joblib.load(filename)
        print(f"✅ Modelo '{filename}' cargado exitosamente.")
    except FileNotFoundError:
        print(f"❌ ADVERTENCIA: No se encontró el archivo del modelo '{filename}'.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo '{filename}': {e}")

# --- FUNCIÓN INTELIGENTE PARA PREPARAR DATOS ---
def prepare_features_from_live_weather(lat, lng):
    """
    Obtiene el clima actual de OpenWeatherMap y simula los datos históricos
    que nuestro modelo necesita para funcionar.
    """
    if not OPENWEATHERMAP_API_KEY or OPENWEATHERMAP_API_KEY == "PEGA_AQUI_TU_API_KEY_DE_OPENWEATHERMAP":
        print("❌ Error: La API Key de OpenWeatherMap no está configurada.")
        return None

    try:
        # 1. Llamar a OpenWeatherMap para obtener el clima actual
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error al llamar a OpenWeatherMap: {response.text}")
            return None
        
        live_data = response.json()
        
        # 2. Extraer los datos actuales como base para la simulación
        current_temp = live_data['main']['temp']
        current_humidity = live_data['main']['humidity']
        current_pressure = live_data['main']['pressure'] / 10 # Convertir hPa a kPa para que coincida con tus datos de entrenamiento
        current_wind = live_data['wind']['speed']

        # 3. Simular los datos de los últimos 6 meses (lags)
        features = {}
        for lag in range(1, 7):
            # Simular una pequeña variación para cada mes anterior
            temp_variation = np.random.uniform(-3, 3) - lag * 0.5
            hum_variation = np.random.uniform(-10, 10) - lag
            press_variation = np.random.uniform(-0.5, 0.5)
            wind_variation = np.random.uniform(-1, 1)

            features[f'T2M_lag_{lag}'] = current_temp + temp_variation
            features[f'RH2M_lag_{lag}'] = np.clip(current_humidity + hum_variation, 20, 100)
            features[f'PS_lag_{lag}'] = current_pressure + press_variation
            features[f'WS2M_lag_{lag}'] = max(0, current_wind + wind_variation)
        
        features['month'] = pd.Timestamp.now().month
        
        return features

    except Exception as e:
        print(f"Error preparando las características (features): {e}")
        return None

# --- RUTAS (ENDPOINTS) DE LA API ---

@app.route('/')
def home():
    """Endpoint para verificar que la API está funcionando."""
    return jsonify({
        'message': 'API de clima inteligente para Lima está en línea',
        'models_loaded_successfully': list(MODELS.keys())
    })

@app.route('/predict', methods=['GET'])
def predict():
    """Recibe coordenadas, obtiene datos en vivo, simula el historial y devuelve la predicción."""
    if not MODELS:
        return jsonify({'error': 'Ningún modelo de predicción está disponible.'}), 500

    # Obtener lat y lng de los parámetros de la URL
    lat = request.args.get('lat')
    lng = request.args.get('lng')

    if not lat or not lng:
        return jsonify({'error': 'Los parámetros "lat" y "lng" son requeridos en la URL.'}), 400

    # 1. Preparar los features usando la función auxiliar
    features = prepare_features_from_live_weather(float(lat), float(lng))
    if features is None:
        return jsonify({'error': 'No se pudieron obtener o simular los datos históricos. Revisa la API Key de OpenWeatherMap.'}), 500
    
    try:
        # 2. Preparar el DataFrame para los modelos
        input_df = pd.DataFrame(features, index=[0])

        # 3. Realizar una predicción con cada modelo
        predictions = {}
        for var, model in MODELS.items():
            prediction = model.predict(input_df)
            predictions[var] = round(float(prediction[0]), 2)
        
        # 4. Devolver la respuesta completa y útil
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