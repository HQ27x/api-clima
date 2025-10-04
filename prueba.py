# test_api.py

import requests
import json

# 1. La URL de tu API en Render (asegúrate de que termine en /predict)
api_url = "https://api-clima-scavengers.onrender.com/predict"

# 2. Los datos de entrada en formato de diccionario de Python
# Deben coincidir exactamente con las características que espera tu modelo.
input_data = {
    "T2M_lag_1": 18.5, "RH2M_lag_1": 85.1, "PS_lag_1": 101.2,
    "T2M_lag_2": 19.1, "RH2M_lag_2": 86.2, "PS_lag_2": 101.3,
    "T2M_lag_3": 19.8, "RH2M_lag_3": 87.0, "PS_lag_3": 101.4,
    "T2M_lag_4": 20.5, "RH2M_lag_4": 85.5, "PS_lag_4": 101.5,
    "T2M_lag_5": 21.2, "RH2M_lag_5": 84.9, "PS_lag_5": 101.4,
    "T2M_lag_6": 22.0, "RH2M_lag_6": 83.1, "PS_lag_6": 101.2,
    "month": 8,
    "year": 2024
}

print(f"🚀 Enviando datos a la API: {api_url}")

try:
    # 3. Realizar la petición POST a la API
    # El argumento `json=input_data` convierte automáticamente el diccionario a JSON
    response = requests.post(api_url, json=input_data)

    # 4. Comprobar el resultado
    if response.status_code == 200:
        # Si la respuesta es exitosa (código 200), la procesamos
        prediction = response.json()
        temp = prediction.get('predicted_temperature')
        print("\n✅ ¡Respuesta recibida exitosamente!")
        print(f"   🌡️ Predicción de temperatura: {temp:.2f} °C")
    else:
        # Si hay un error, mostramos el código y el mensaje
        print("\n❌ Error al contactar la API.")
        print(f"   Código de estado: {response.status_code}")
        print(f"   Respuesta del servidor: {response.text}")

except requests.exceptions.RequestException as e:
    # Capturar errores de conexión (ej. no hay internet)
    print(f"\n❌ Error de conexión: {e}")