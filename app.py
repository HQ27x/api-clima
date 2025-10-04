
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# --- CONFIGURACIÓN ---
app = Flask(__name__)
CORS(app) # Habilita CORS

# --- CARGA DEL MODELO ---
model = None
try:
    # Render busca los archivos en la raíz del proyecto
    model = joblib.load('weather_model.joblib')
    print("✅ Modelo 'weather_model.joblib' cargado exitosamente.")
except Exception as e:
    print(f"❌ Error crítico: No se pudo cargar el modelo. Error: {e}")

# --- ENDPOINTS (RUTAS) DE LA API ---

@app.route('/')
def home():
    """Ruta para verificar que la API está funcionando."""
    status = "cargado exitosamente" if model is not None else "falló al cargar"
    return jsonify({
        'message': 'API del modelo de clima está en línea',
        'model_status': status
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Ruta para realizar las predicciones."""
    if model is None:
        return jsonify({'error': 'El modelo no está disponible.'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No se recibieron datos en el cuerpo (body) de la petición.'}), 400

    try:
        # Convertir los datos recibidos a un DataFrame de Pandas
        input_df = pd.DataFrame(data, index=[0])
        
        # Realizar la predicción
        prediction = model.predict(input_df)
        
        # Devolver el resultado
        return jsonify({'predicted_temperature': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': f'Ocurrió un error en la predicción: {str(e)}'}), 500

# Esta parte es para pruebas locales, Render no la usa directamente
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
