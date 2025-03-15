import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Cargue del modelo

def load_model(model_path):
    """Carga un modelo de Keras guardado en formato .keras o .h5."""
    return tf.keras.models.load_model(model_path)

# Normalización de data.

def normalize_data(X, scaler_path="models/scaler_X.pkl"):
    """Normaliza los datos de entrada usando un scaler guardado."""
    scaler = joblib.load(scaler_path)
    return scaler.transform(X)

# Desnormalización de predicciones. 

def denormalize_data(y, scaler_path="models/scaler_y.pkl"):
    """Desnormaliza los datos de salida usando un scaler guardado."""
    scaler = joblib.load(scaler_path)
    return scaler.inverse_transform(y.reshape(-1, 1)).flatten()

# Guardando el scaler
def save_scaler(scaler, path):
    """Guarda un scaler en un archivo para reutilización."""
    joblib.dump(scaler, path)


# Aquí probamos guardando el modelo. 
if __name__ == "__main__":
    model = load_model("models/rental_price_model.keras")
    print("Modelo cargado correctamente.")
