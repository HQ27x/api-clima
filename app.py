# app.py - Versión corregida y simplificada para tu modelo

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# --- CONFIGURACIÓN ---
app = Flask(__name__)
CORS(app) # Habilita CORS para permitir peticiones desde tu web

# --- CARGA DEL MODELO ---
model = None
try:
    # Render busca los archivos en la raíz del proyecto
    model = joblib.load('weather_model.joblib')
    print("✅ Modelo cargado exitosamente.")
except Exception as e:
    print(f"❌ Error crítico: No se pudo cargar el modelo 'weather_model.joblib'. Error: {e}")

# --- ENDPOINTS DE LA API ---

@app.route('/')
def home():
    """Endpoint para verificar que la API está funcionando."""
    status = "cargado exitosamente" if model is not None else "falló al cargar"
    return jsonify({
        'message': 'API del modelo de clima está en línea',
        'model_status': status
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones."""
    if model is None:
        return jsonify({'error': 'El modelo no está disponible, revisa los logs del servidor.'}), 500

    # Obtener los datos JSON de la petición
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No se recibieron datos en el cuerpo de la petición (body).'}), 400

    try:
        # Convertir los datos recibidos a un DataFrame de Pandas
        # El modelo espera exactamente este formato
        input_df = pd.DataFrame(data, index=[0])

        # Realizar la predicción
        prediction = model.predict(input_df)

        # Devolver el resultado en formato JSON
        return jsonify({'predicted_temperature': float(prediction[0])})

    except KeyError as e:
        return jsonify({'error': f'Falta la siguiente característica en los datos de entrada: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'Ocurrió un error durante la predicción: {str(e)}'}), 500

# Esta parte no es necesaria para Render, pero es buena práctica
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)