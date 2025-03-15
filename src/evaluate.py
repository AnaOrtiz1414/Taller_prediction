import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_data
from utils import load_model, denormalize_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# datos y modelo
file_path = r"C:\Users\ANA\Documents\Maestria\Clase Deeplearning\Taller entregable\archive\House_Rent_Dataset.csv"
X_train, X_test, y_train, y_test = load_data(file_path)
model = load_model("models/rental_price_model.keras")

#  predicciones
y_pred = model.predict(X_test)

#  Desnormalizar los datos
y_test_original = denormalize_data(y_test)
y_pred_original = denormalize_data(y_pred)

#  métricas de evaluación
mae = mean_absolute_error(y_test_original, y_pred_original)
mse = mean_squared_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

#  Predicciones vs. Valores reales
plt.figure(figsize=(8,5))
plt.scatter(y_test_original, y_pred_original, alpha=0.5)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Predicciones vs. Valores Reales")
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], linestyle='--', color='red')
plt.show()
