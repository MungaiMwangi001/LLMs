from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble  import RandomForestClassifier
import joblib


#loa d the iris dataset
iris  = load_iris()
x, y = iris.data, iris.target   

#split the dataset into training and testing sets
x_train, x_test , y_train , y_test = train_test_split(x, y ,test_size = 0.3, random_state= 42)

# train the  model

model =  RandomForestClassifier()
model.fit(x_train, y_train)


#save_model
joblib.dump(model, "model.pkl")

print("model trained an saved as model.pkl succesfully")