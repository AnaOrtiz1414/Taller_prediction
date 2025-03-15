import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file_path):
    """Carga los datos desde un archivo CSV y realiza preprocesamiento."""
    df = pd.read_csv(file_path)
    
    #Verificar las columnas disponibles
    print("Columnas disponibles en el dataset:", df.columns.tolist())
    
    # Selección de características y variable objetivo
    features = ["Size", "BHK", "Bathroom"]
    target = "Rent"
    
    # Verificar que las columnas existen
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Las siguientes columnas faltan en el dataset: {missing_cols}")
    
    df = df[features + [target]]
    
    # Separar datos
    y = df[target].values
    X = df.drop(columns=[target]).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizar datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    file_path = r"C:\Users\ANA\Documents\Maestria\Clase Deeplearning\Taller entregable\archive\House_Rent_Dataset.csv"
    X_train, X_test, y_train, y_test = load_data(file_path)
    print("Datos cargados y preprocesados correctamente.")
