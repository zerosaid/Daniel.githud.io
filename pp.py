import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Leer el archivo CSV
df = pd.read_csv("cbb.csv")

# Usar LabelEncoder para convertir "TEAM" (asumiendo que es una columna categórica) a valores numéricos
encoder = LabelEncoder()
df["TEAM_encoded"] = encoder.fit_transform(df["TEAM"])

# Definir las variables independientes (X) y dependientes (y)
x = df[["TEAM_encoded"]]  # Asegurarse de que x sea 2D
y = df[["G"]]  # Esto ya es 2D, por lo que no hay problema

# Dividir el conjunto de datos en entrenamiento y validación
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

# Crear y entrenar el modelo de regresión (puedes usar también LinearRegression si lo prefieres)
model = DecisionTreeRegressor(random_state=1)
model.fit(train_x, train_y)

# Hacer predicciones con el modelo
prediccion_futura = model.predict(val_x)

# Graficar los datos de entrenamiento, validación y predicciones
plt.scatter(train_x, train_y, color='blue', label='Train Data')
plt.scatter(val_x, val_y, color='green', label='Validation Data')
plt.scatter(val_x, prediccion_futura, color='red', label='Predictions')
plt.xlabel('Team Encoded')
plt.ylabel('Games (G)')
plt.legend()
plt.show()
