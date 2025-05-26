import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os # Importar el módulo os para manejar rutas de archivos

# --- 1. Configuración de Rutas ---
# Definir la ruta base donde se guardaron el modelo y los escaladores
# Es mejor usar os.path.join para construir rutas de forma compatible con cualquier SO (Windows/Linux)
BASE_MODEL_PATH = r"D:\1. Proyectos\Red Neuronal\Prediccion de costo de viviendas\Modelos"

# Definir los nombres específicos de los archivos (deben coincidir con los de guardado)
MODEL_NAME = "modelo_precio_casas.keras" # Asegúrate que este nombre coincida con el de tu script de entrenamiento
X_SCALER_NAME = "scaler_X.pkl"
Y_SCALER_NAME = "scaler_y.pkl"

# --- 2. Carga de Modelo y Escaladores ---
def load_artifacts(base_path, model_name, x_scaler_name, y_scaler_name):
    """
    Carga el modelo de Keras y los objetos StandardScaler guardados.
    Maneja errores si los archivos no se encuentran.
    """
    try:
        model_path = os.path.join(base_path, model_name)
        x_scaler_path = os.path.join(base_path, x_scaler_name)
        y_scaler_path = os.path.join(base_path, y_scaler_name)

        modelo_cargado = load_model(model_path)
        scaler_X_cargado = joblib.load(x_scaler_path)
        scaler_y_cargado = joblib.load(y_scaler_path)

        print(f"✅ Modelo '{model_name}' cargado exitosamente.")
        print(f"✅ Escalador de características '{x_scaler_name}' cargado exitosamente.")
        print(f"✅ Escalador de precios '{y_scaler_name}' cargado exitosamente.")
        return modelo_cargado, scaler_X_cargado, scaler_y_cargado

    except FileNotFoundError:
        print(f"❌ Error: Uno o más archivos no se encontraron en '{base_path}'.")
        print("Asegúrate de que los nombres y la ruta sean correctos y que los archivos existan.")
        return None, None, None
    except Exception as e:
        print(f"❌ Ocurrió un error al cargar los artefactos del modelo: {e}")
        return None, None, None

# Cargar los componentes del modelo al inicio del script
modelo, scaler_X, scaler_y = load_artifacts(BASE_MODEL_PATH, MODEL_NAME, X_SCALER_NAME, Y_SCALER_NAME)

# Si la carga falla, salir del script
if modelo is None or scaler_X is None or scaler_y is None:
    exit() # Termina la ejecución si no se pueden cargar los componentes

# --- 3. Función para Realizar Predicciones ---
def predict_house_price(model, scaler_X, scaler_y):
    """
    Solicita al usuario las características de una casa,
    realiza la predicción y la muestra.
    """
    print("\n--- ¡Hagamos una Predicción de Precio de Vivienda! ---")
    print("Por favor, ingrese los siguientes datos para la nueva casa:")

    try:
        # Validar entrada para Lot Area (solo números positivos)
        while True:
            try:
                lot_area = float(input("Área del terreno (Lot Area en pies cuadrados): "))
                if lot_area >= 0:
                    break
                else:
                    print("Por favor, ingrese un valor positivo para el Área del terreno.")
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número.")

        # Validar entrada para Garage Area (solo números positivos)
        while True:
            try:
                garage_area = float(input("Área del garaje (Garage Area en pies cuadrados): "))
                if garage_area >= 0:
                    break
                else:
                    print("Por favor, ingrese un valor positivo para el Área del garaje.")
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número.")

        # Validar entrada para Garage Cars (número entero positivo)
        while True:
            try:
                garage_cars = int(input("Número de autos en garaje (Garage Cars, entero): "))
                if garage_cars >= 0:
                    break
                else:
                    print("Por favor, ingrese un número entero positivo para el número de autos.")
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número entero.")

        # Validar entrada para Year Built (año válido)
        # Puedes añadir más validación, como un rango de años (ej. 1800-2025)
        while True:
            try:
                year_built = int(input("Año de construcción (Year Built, ej. 2005): "))
                if 1800 <= year_built <= 2025: # Rango razonable de años
                    break
                else:
                    print("Por favor, ingrese un año de construcción válido (ej. entre 1800 y 2025).")
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número entero.")

        while True: 
            try:
                overall_qual = int(input("Calidad general de la casa (Overall Qual): "))
                if overall_qual >= 0:
                      break
                else:
                   print("Por favor, ingrese un valor positivo para la calidad de la casa (ej, 1 y 10).")
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número.")
        while True: 
            try:
                gr_liv_area = float(input("Área habitable sobre suelo (Gr Liv Area): "))
                if gr_liv_area >= 0:
                      break
                else:
                   print("Por favor, ingrese un valor positivo para el Área habitable sobre suelo.")
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número.")
        while True:
            try:
                total_bsmt_sf = float(input("Área total del sótano (Total Bsmt SF): "))
                if total_bsmt_sf >= 0:
                      break
                else:
                   print("Por favor, ingrese un valor positivo para el Área total del sótano.")
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número.")
        while True:
            try:
                first_flr_sf = float(input("Área del primer piso (1st Flr SF): "))
                if first_flr_sf >= 0:
                      break
                else:
                   print("Por favor, ingrese un valor positivo para el Área del primer piso.")
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número.")


        # Crear arreglo con los datos ingresados (asegurándose que sea un array 2D para el escalador)
        nueva_casa = np.array([[lot_area, garage_area, garage_cars, year_built, overall_qual, gr_liv_area, total_bsmt_sf, first_flr_sf]]) 

        # Escalar las nuevas características usando el scaler_X entrenado
        nueva_casa_scaled = scaler_X.transform(nueva_casa)

        # Realizar la predicción con el modelo
        pred_scaled = model.predict(nueva_casa_scaled, verbose=0) # verbose=0 para no mostrar barra de progreso

        # Desescalar la predicción para obtener el precio real
        pred_real = scaler_y.inverse_transform(pred_scaled)

        # Imprimir el resultado formateado
        # [0][0] porque pred_real es un array 2D (ej. [[178930.81]])
        print(f"\n✅ El precio estimado de la vivienda es: ${pred_real[0][0]:,.2f}")
        print("--------------------------------------------------")

    except ValueError:
        print("❌ Error de entrada: Asegúrese de ingresar números válidos para los campos.")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado durante la predicción: {e}")

# --- Ejecutar la función de predicción ---
if __name__ == "__main__":
    predict_house_price(modelo, scaler_X, scaler_y)