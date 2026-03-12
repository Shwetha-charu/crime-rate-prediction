import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

data = pd.read_csv("crime_data.csv")

X = data[['population','unemployment','poverty','police']]
y = data['crime_rate']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

pickle.dump(model, open("model.pkl","wb"))

print("Model trained successfully")