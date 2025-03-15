import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_shape):
    """Define una red neuronal simple en Keras Sequential."""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  
    ])
    
    # Compilar el modelo
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

#Probando el modelo

if __name__ == "__main__":
    sample_input_shape = 3 
    model = create_model(sample_input_shape)
    model.summary()
