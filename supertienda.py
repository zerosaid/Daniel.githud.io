
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("cbb.csv")
plt = pd.read_csv("cbb.csv")
sns = pd.read_csv("cbb.csv")

x = df["TEAM"]
y = df[["G"]] 

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1) 
model=DecisionTreeRegressor(random_state=1)
model.fit(train_x,train_y)

prediccion_futura = model.predict(val_x)

plt.scatter(train_x, train_y)
plt.scatter(val_x, val_y)
plt.scatter(val_x, prediccion_futura)
plt.legend()
plt.show()

