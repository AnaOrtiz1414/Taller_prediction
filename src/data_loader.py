import pandas as pd

def load_data(file_path):
    """Carga los datos desde un archivo CSV y devuelve un DataFrame."""
    df = pd.read_csv(file_path)
    print("Datos cargados con éxito. Primeras filas:")
    print(df.head())  # Muestra las primeras filas
    return df

if __name__ == "__main__":
    file_path = r"C:\Users\ANA\Documents\Maestria\Clase Deeplearning\Taller entregable\archive\House_Rent_Dataset.csv"  
    df = load_data(file_path)
