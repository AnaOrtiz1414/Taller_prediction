import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_shape):
    """Define una red neuronal en Keras Sequential."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),  # Más neuronas
        layers.BatchNormalization(),  # Normalización para estabilidad
        layers.Dropout(0.3),  # Regularización para evitar overfitting
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),

        layers.Dense(1)  # Capa de salida sin activación para regresión
    ])
    
    # Compilar el modelo con un optimizador más refinado
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    
    return model

# Probando el modelo
if __name__ == "__main__":
    sample_input_shape = 3  # Ajusta este valor según la cantidad de features
    model = create_model(sample_input_shape)
    model.summary()
