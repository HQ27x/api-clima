"""
Weather NASA API - Modelo de Machine Learning para predicción del tiempo
Optimizada para deployment en Render.com
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Permitir CORS para cualquier dominio

# Variables globales
model = None
feature_names = [
    'T2M_lag_1', 'RH2M_lag_1', 'PS_lag_1',
    'T2M_lag_2', 'RH2M_lag_2', 'PS_lag_2',
    'T2M_lag_3', 'RH2M_lag_3', 'PS_lag_3',
    'T2M_lag_4', 'RH2M_lag_4', 'PS_lag_4',
    'T2M_lag_5', 'RH2M_lag_5', 'PS_lag_5',
    'T2M_lag_6', 'RH2M_lag_6', 'PS_lag_6',
    'month', 'year'
]

def load_model():
    """Cargar el modelo de Machine Learning"""
    global model
    try:
        # Buscar el modelo en diferentes ubicaciones
        model_paths = [
            'weather_model.joblib',
            'models/weather_model.joblib',
            '/app/weather_model.joblib',  # Para Render
            '/opt/render/project/src/weather_model.joblib'  # Para Render
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                logger.info(f"✅ Modelo cargado desde: {path}")
                return True
        
        logger.error("❌ No se encontró el modelo en ninguna ubicación")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error al cargar modelo: {e}")
        return False

def prepare_features(lat, lng):
    """Preparar características para el modelo"""
    try:
        # Simular datos históricos basados en ubicación geográfica
        features = {}
        
        # Generar datos lagged (T2M, RH2M, PS) para los últimos 6 días
        for i in range(1, 7):
            # Temperatura (T2M) - basada en latitud y estación
            base_temp = 25 - (lat * 0.5)  # Más frío hacia el sur
            seasonal = np.sin((datetime.now().month - 1) * np.pi / 6) * 10
            noise = np.random.uniform(-5, 5)
            lag_effect = i * 0.1
            
            features[f'T2M_lag_{i}'] = base_temp + seasonal + noise + lag_effect
            
            # Humedad (RH2M) - 40-90%
            base_humidity = 60 + (lat * 0.2)
            noise = np.random.uniform(-20, 30)
            lag_effect = i * 0.5
            
            features[f'RH2M_lag_{i}'] = np.clip(base_humidity + noise + lag_effect, 40, 90)
            
            # Presión (PS) - 950-1050 hPa
            base_pressure = 1013 + (lat * 0.1)
            noise = np.random.uniform(-50, 50)
            lag_effect = i * 0.2
            
            features[f'PS_lag_{i}'] = np.clip(base_pressure + noise + lag_effect, 950, 1050)
        
        # Mes y año actual
        features['month'] = datetime.now().month
        features['year'] = datetime.now().year
        
        return features
        
    except Exception as e:
        logger.error(f"❌ Error preparando características: {e}")
        return None

@app.route('/')
def home():
    """Endpoint de bienvenida"""
    return jsonify({
        'message': '🌦️ NASA Weather Prediction API',
        'version': '1.0.0',
        'status': 'active',
        'model_loaded': model is not None,
        'endpoints': {
            'predict': '/predict',
            'health': '/health',
            'model_info': '/model_info',
            'cities': '/cities'
        },
        'documentation': 'https://github.com/tu-usuario/weather-nasa-api',
        'author': 'NASA Weather Prediction Team'
    })

@app.route('/health')
def health():
    """Health check para Render"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'uptime': 'active',
        'version': '1.0.0'
    })

@app.route('/model_info')
def model_info():
    """Información del modelo"""
    return jsonify({
        'model_type': 'NASA Weather Prediction Model',
        'features': feature_names,
        'n_features': len(feature_names),
        'version': '1.0.0',
        'description': 'Modelo de Machine Learning para predicción de temperatura promedio mensual',
        'loaded': model is not None,
        'input_description': {
            'T2M_lag_X': 'Temperatura a 2 metros (lag X días)',
            'RH2M_lag_X': 'Humedad relativa a 2 metros (lag X días)',
            'PS_lag_X': 'Presión superficial (lag X días)',
            'month': 'Mes del año (1-12)',
            'year': 'Año'
        },
        'output_description': {
            'temperature_avg': 'Temperatura promedio mensual en °C',
            'confidence': 'Nivel de confianza de la predicción (0-1)',
            'month_name': 'Nombre del mes en inglés',
            'year': 'Año de la predicción'
        }
    })

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    """Predicción principal del modelo"""
    try:
        # Obtener datos de entrada
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args
        
        lat = float(data.get('lat', 0))
        lng = float(data.get('lng', 0))
        
        if not lat or not lng:
            return jsonify({
                'success': False,
                'error': 'Coordenadas lat y lng requeridas'
            }), 400
        
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Modelo no cargado'
            }), 500
        
        # Preparar características
        features = prepare_features(lat, lng)
        if features is None:
            return jsonify({
                'success': False,
                'error': 'Error preparando características'
            }), 500
        
        # Crear DataFrame con las características en el orden correcto
        feature_array = [features[name] for name in feature_names]
        X = pd.DataFrame([feature_array], columns=feature_names)
        
        # Hacer predicción
        prediction = model.predict(X)[0]
        
        # Limitar predicción entre 0-50°C
        prediction = np.clip(prediction, 0, 50)
        
        return jsonify({
            'success': True,
            'prediction': {
                'temperature_avg': round(float(prediction), 2),
                'month_name': datetime.now().strftime('%B'),
                'year': datetime.now().year,
                'confidence': 0.85,
                'model_type': 'NASA Model (API)',
                'features_used': len(feature_names),
                'location': {
                    'lat': lat,
                    'lng': lng
                },
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return jsonify({
            'success': False,
            'error': f'Error en predicción: {str(e)}'
        }), 500

@app.route('/cities', methods=['POST', 'GET'])
def cities():
    """Buscar ciudades por coordenadas"""
    try:
        # Obtener datos de entrada
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args
        
        lat = float(data.get('lat', 0))
        lng = float(data.get('lng', 0))
        
        if not lat or not lng:
            return jsonify({
                'success': False,
                'error': 'Coordenadas lat y lng requeridas'
            }), 400
        
        # Base de datos de ciudades de América del Sur
        cities = [
            # Brasil
            {'nome': 'São Paulo', 'uf': 'SP', 'lat': -23.5505, 'lng': -46.6333, 'pais': 'Brasil'},
            {'nome': 'Rio de Janeiro', 'uf': 'RJ', 'lat': -22.9068, 'lng': -43.1729, 'pais': 'Brasil'},
            {'nome': 'Brasília', 'uf': 'DF', 'lat': -15.7801, 'lng': -47.9292, 'pais': 'Brasil'},
            {'nome': 'Salvador', 'uf': 'BA', 'lat': -12.9777, 'lng': -38.5016, 'pais': 'Brasil'},
            {'nome': 'Fortaleza', 'uf': 'CE', 'lat': -3.7319, 'lng': -38.5267, 'pais': 'Brasil'},
            {'nome': 'Belo Horizonte', 'uf': 'MG', 'lat': -19.9167, 'lng': -43.9345, 'pais': 'Brasil'},
            {'nome': 'Manaus', 'uf': 'AM', 'lat': -3.1190, 'lng': -60.0217, 'pais': 'Brasil'},
            {'nome': 'Curitiba', 'uf': 'PR', 'lat': -25.4244, 'lng': -49.2654, 'pais': 'Brasil'},
            {'nome': 'Recife', 'uf': 'PE', 'lat': -8.0476, 'lng': -34.8770, 'pais': 'Brasil'},
            {'nome': 'Porto Alegre', 'uf': 'RS', 'lat': -30.0346, 'lng': -51.2177, 'pais': 'Brasil'},
            
            # Perú
            {'nome': 'Lima', 'uf': 'LIM', 'lat': -12.0464, 'lng': -77.0428, 'pais': 'Perú'},
            {'nome': 'Arequipa', 'uf': 'ARE', 'lat': -16.4090, 'lng': -71.5375, 'pais': 'Perú'},
            {'nome': 'Trujillo', 'uf': 'TRU', 'lat': -8.1116, 'lng': -79.0288, 'pais': 'Perú'},
            {'nome': 'Chiclayo', 'uf': 'CHI', 'lat': -6.7714, 'lng': -79.8409, 'pais': 'Perú'},
            {'nome': 'Cusco', 'uf': 'CUS', 'lat': -13.5319, 'lng': -71.9675, 'pais': 'Perú'},
            
            # Argentina
            {'nome': 'Buenos Aires', 'uf': 'BA', 'lat': -34.6118, 'lng': -58.3960, 'pais': 'Argentina'},
            {'nome': 'Córdoba', 'uf': 'CB', 'lat': -31.4201, 'lng': -64.1888, 'pais': 'Argentina'},
            {'nome': 'Rosario', 'uf': 'SF', 'lat': -32.9442, 'lng': -60.6505, 'pais': 'Argentina'},
            {'nome': 'Mendoza', 'uf': 'MZ', 'lat': -32.8908, 'lng': -68.8272, 'pais': 'Argentina'},
            
            # Chile
            {'nome': 'Santiago', 'uf': 'RM', 'lat': -33.4489, 'lng': -70.6693, 'pais': 'Chile'},
            {'nome': 'Valparaíso', 'uf': 'VS', 'lat': -33.0458, 'lng': -71.6197, 'pais': 'Chile'},
            {'nome': 'Concepción', 'uf': 'BI', 'lat': -36.8201, 'lng': -73.0444, 'pais': 'Chile'},
            
            # Colombia
            {'nome': 'Bogotá', 'uf': 'DC', 'lat': 4.7110, 'lng': -74.0721, 'pais': 'Colombia'},
            {'nome': 'Medellín', 'uf': 'AN', 'lat': 6.2442, 'lng': -75.5812, 'pais': 'Colombia'},
            {'nome': 'Cali', 'uf': 'CA', 'lat': 3.4516, 'lng': -76.5320, 'pais': 'Colombia'},
            
            # Ecuador
            {'nome': 'Quito', 'uf': 'EC', 'lat': -0.1807, 'lng': -78.4678, 'pais': 'Ecuador'},
            {'nome': 'Guayaquil', 'uf': 'EC', 'lat': -2.1894, 'lng': -79.8890, 'pais': 'Ecuador'},
            
            # Bolivia
            {'nome': 'La Paz', 'uf': 'LP', 'lat': -16.5000, 'lng': -68.1500, 'pais': 'Bolivia'},
            {'nome': 'Santa Cruz', 'uf': 'SC', 'lat': -17.7833, 'lng': -63.1833, 'pais': 'Bolivia'},
            
            # Uruguay
            {'nome': 'Montevideo', 'uf': 'MO', 'lat': -34.9011, 'lng': -56.1645, 'pais': 'Uruguay'},
            
            # Paraguay
            {'nome': 'Asunción', 'uf': 'AS', 'lat': -25.2637, 'lng': -57.5759, 'pais': 'Paraguay'},
            
            # Venezuela
            {'nome': 'Caracas', 'uf': 'DC', 'lat': 10.4806, 'lng': -66.9036, 'pais': 'Venezuela'},
            {'nome': 'Maracaibo', 'uf': 'ZU', 'lat': 10.6666, 'lng': -71.6124, 'pais': 'Venezuela'}
        ]
        
        # Encontrar ciudad más cercana
        closest_city = None
        min_distance = float('inf')
        
        for city in cities:
            distance = np.sqrt((city['lat'] - lat)**2 + (city['lng'] - lng)**2)
            if distance < min_distance:
                min_distance = distance
                closest_city = city
        
        if closest_city and min_distance < 0.5:  # Dentro de ~50km
            closest_city['id'] = f"sim_{closest_city['nome'].lower().replace(' ', '_')}"
            closest_city['distancia_km'] = round(min_distance * 111, 1)
            
            return jsonify({
                'success': True,
                'data': closest_city
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No se encontró una ciudad cercana (distancia mínima: {round(min_distance * 111, 1)}km)'
            }), 404
        
    except Exception as e:
        logger.error(f"Error en búsqueda de ciudades: {e}")
        return jsonify({
            'success': False,
            'error': f'Error en búsqueda: {str(e)}'
        }), 500

@app.route('/forecast', methods=['POST', 'GET'])
def forecast():
    """Pronóstico del tiempo simulado"""
    try:
        # Obtener datos de entrada
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args
        
        city_id = data.get('city_id')
        city_name = data.get('city_name', 'Ciudad')
        pais = data.get('pais', 'Brasil')
        
        if not city_id:
            return jsonify({
                'success': False,
                'error': 'ID de ciudad requerido'
            }), 400
        
        # Generar pronóstico simulado
        forecast = []
        for i in range(5):
            date = datetime.now().replace(day=datetime.now().day + i)
            
            temp_base = 25
            if pais == 'Perú': temp_base = 22
            elif pais == 'Argentina': temp_base = 20
            elif pais == 'Chile': temp_base = 18
            elif pais == 'Colombia': temp_base = 28
            elif pais == 'Ecuador': temp_base = 26
            elif pais == 'Bolivia': temp_base = 19
            elif pais == 'Uruguay': temp_base = 21
            elif pais == 'Paraguay': temp_base = 27
            elif pais == 'Venezuela': temp_base = 29
            
            temp_min = temp_base + np.random.uniform(-5, 0)
            temp_max = temp_base + np.random.uniform(0, 8)
            
            conditions = ['Parcialmente nublado', 'Soleado', 'Nublado', 'Lluvia ligera', 'Despejado']
            
            forecast.append({
                'data': date.strftime('%Y-%m-%d'),
                'condicao_desc': np.random.choice(conditions),
                'min': round(temp_min, 1),
                'max': round(temp_max, 1),
                'indice_uv': np.random.randint(3, 11)
            })
        
        result = {
            'cidade': f"{city_name}, {pais}",
            'clima': forecast
        }
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error en pronóstico: {e}")
        return jsonify({
            'success': False,
            'error': f'Error en pronóstico: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint no encontrado',
        'available_endpoints': ['/', '/health', '/model_info', '/predict', '/cities', '/forecast']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Error interno del servidor'
    }), 500

if __name__ == '__main__':
    # Cargar modelo al iniciar
    load_model()
    
    # Obtener puerto del entorno (para Render)
    port = int(os.environ.get('PORT', 5000))
    
    # Ejecutar aplicación
    app.run(host='0.0.0.0', port=port, debug=False)
