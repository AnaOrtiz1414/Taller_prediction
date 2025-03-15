import tensorflow as tf
import numpy as np
import pandas as pd
from data_loader import load_data


model = tf.keras.models.load_model(
    "models/rental_price_model.h5",
    compile=False  # Evita errores de compilación al cargar
)
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])



# simulación 

nuevos_datos = np.array([[1200, 3, 2]])  # (Size, BHK, Bathroom)

# Preprocesamiento de datos 

file_path = r"C:\Users\ANA\Documents\Maestria\Clase Deeplearning\Taller entregable\archive\House_Rent_Dataset.csv"
X_train, X_test, y_train, y_test = load_data(file_path)
mean, std = X_train.mean(axis=0), X_train.std(axis=0)  # Obtener estadísticas de entrenamiento
nuevos_datos = (nuevos_datos - mean) / std  # Normalizar

print(f"Número de características usadas en el modelo: {X_train.shape[1]}")


# Predicción

y_pred = model.predict(nuevos_datos)
print(f"Predicción de precio de alquiler: {y_pred[0][0]:.2f}")