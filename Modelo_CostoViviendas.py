import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib 
from pathlib import Path


# Cargar datos
base = Path(__file__).parent

# Ruta completa al archivo CSV
ruta_arch = base / "Datos" / "AmesHousing.csv"

try:
    data = pd.read_csv(ruta_arch, encoding="utf-8")
    print("Archivo encontrado exitosamente:\n", data.head(10))
except:
    print("Archivo no encontrado, intentalo de nuevo")
    exit()

# Mostrar nombres de las columnas
print("Nombre de las columnas:", data.columns)


#----------------------Preparamos los datos------------------------


# Selección de características y objetivo
features = ["Lot Area", "Garage Area", "Garage Cars", "Year Built", 
          "Overall Qual", "Gr Liv Area", "Total Bsmt SF", "1st Flr SF"]


X = data[features].dropna()
y = data.loc[X.index, "SalePrice"]

# División y escalado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))


# -----------------Construccion del modelo-------------
import tensorflow as tf

# Crear el modelo de red neuronal


modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

#Copilacion del modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss="mean_squared_error",
    metrics=["mae"]
)

print("----------------Entrenar modelo----------")
historial = modelo.fit(X_train_scaled, y_train_scaled, epochs=100, verbose=1)
print("¡Modelo entrenado!")

#--------------------VISUALIZACIÓN DEL ENTRENAMIENTO---------------

plt.plot(historial.history['loss'], label='Pérdida (MSE)')
plt.plot(historial.history['mae'], label='Error absoluto medio (MAE)')
plt.title('Evolución del entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()


# Evaluación en el conjunto de prueba
loss, mae = modelo.evaluate(X_test_scaled, y_scaler.transform(y_test.values.reshape(-1, 1)))
print(f"\n Evaluación del modelo:")
print(f"  - Error cuadrático medio (MSE): {loss}")
print(f"  - Error absoluto medio (MAE): {mae}")


# Realizar predicciones
predicciones = modelo.predict(X_test_scaled)
predicciones_desescaladas = y_scaler.inverse_transform(predicciones)
print("✅ Predicción ejemplo:", round(predicciones_desescaladas[0][0], 2))

# Desescalar las predicciones para volver a la escala original
predicciones_desescaladas = y_scaler.inverse_transform(predicciones)


# --------------- GUARDAR EL MODELO Y ESCALADORES --------------


# Definir la carpeta donde guardar
# Definir la ruta completa en Windows
ruta_guardado = Path(r"D:\1. Proyectos\Red Neuronal\Prediccion de costo de viviendas\Modelos")

#Creas carpeta sino existe
ruta_guardado.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe

# Guardar modelo y escaladores con ruta completa
modelo.save(ruta_guardado / "modelo_precio_casas.keras")
joblib.dump(scaler, ruta_guardado / "scaler_X.pkl")
joblib.dump(y_scaler, ruta_guardado / "scaler_y.pkl")

print("Modelo y escaladores guardados con éxito en", ruta_guardado.resolve())


"""
# ---------------------- PREDICCIÓN PERSONALIZADA------------------

print("\n--- Predicción personalizada ---")
try:
    print("Por favor, ingresa los datos de la casa:")

    lot_area = float(input("Área del terreno (Lot Area): "))
    garage_area = float(input("Área del garaje (Garage Area): "))
    garage_cars = float(input("Número de autos en garaje (Garage Cars): "))
    year_built = int(input("Año de construcción (Year Built): "))
    overall_qual = int(input("Calidad general de la casa (Overall Qual): "))
    gr_liv_area = float(input("Área habitable sobre suelo (Gr Liv Area): "))
    total_bsmt_sf = float(input("Área total del sótano (Total Bsmt SF): "))
    first_flr_sf = float(input("Área del primer piso (1st Flr SF): "))

    nueva_casa = np.array([[lot_area, garage_area, garage_cars, year_built,
                            overall_qual, gr_liv_area, total_bsmt_sf, first_flr_sf]])

    nueva_casa_scaled = scaler.transform(nueva_casa)

    # Predecir y desescalar
    pred_scaled = modelo.predict(nueva_casa_scaled)
    pred_real = y_scaler.inverse_transform(pred_scaled)

    print(f"\n✅ Precio estimado: ${pred_real[0][0]:,.2f}")

except Exception as e:
    print("⚠️ Error al ingresar los datos:", e)
"""