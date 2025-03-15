import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import load_data
from model import create_model

# Carga de datos. 

file_path = r"C:\Users\ANA\Documents\Maestria\Clase Deeplearning\Taller entregable\archive\House_Rent_Dataset.csv"
X_train, X_test, y_train, y_test = load_data(file_path)

# creación de un modelo

input_shape = X_train.shape[1]
model = create_model(input_shape)

# Entrenamiento del modelo 
epochs = 80
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=32,
    verbose=1
)

# guardando el modelo entrenado

model.save("models/rental_price_model.keras")  # Guardar en formato Keras recomendado
model.save("models/rental_price_model.h5")  # Guardar en formato HDF5
print("Modelo guardado con éxito en 'models/rental_price_model.keras' y 'models/rental_price_model.h5'")


# Visualización de la pérdida

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Loss_train')
plt.plot(history.history['val_loss'], label='Loss_validation')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.title('Evolución de la Pérdida durante el Entrenamiento')
plt.legend()
plt.show()
