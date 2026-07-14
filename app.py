# app.py
import streamlit as st
import joblib
import numpy as np
class Softmax_regression():
  def __init__(self,learning_rate,epoch,batch_sz,num_class):
    self.weights=None
    self.lr=learning_rate
    self.epochs=epoch
    self.batch_size=batch_sz
    self.classes=num_class

  def softmax(self,z):
    # Ensure z is an ndarray to avoid issues with numpy.matrix's sum method
    z = np.asarray(z)
    expZ=np.exp(z)
    return expZ/np.sum(expZ,axis=1,keepdims=True)

  def fit(self,X_train,y_train):
    X_train=np.insert(X_train,0,1,axis=1)
    self.weights=np.ones((self.classes,X_train.shape[1]))

    for i in range(self.epochs):

      idxs = np.random.randint(0, X_train.shape[0]-1, self.batch_size)
      x_train=X_train[idxs]
      z=np.dot(X_train[idxs],self.weights.T)
      y_pred=self.softmax(z)

      #weights gradient
      weights_grad=np.dot((y_pred-y_train[idxs]).T,x_train)
      #updation
      self.weights=self.weights-self.lr*weights_grad

  def predict(self,x_test):
      x_test=np.insert(x_test,0,1,axis=1)
      return np.argmax(np.dot(x_test,self.weights.T),axis=1)

# Load the model
@st.cache_resource  # Keeps the model cached in memory for speed
def load_model():
    model_stuff = joblib.load('model.pkl')
    if isinstance(model_stuff, dict):
        return model_stuff['model']
    return model_stuff

model = load_model()
target_names = ['Setosa', 'Versicolor', 'Virginica']

# 2. Design the Web App Interface
st.title("LogiMax: Flower Classifier 🌸")
st.write("Enter the measurements below to predict the flower species.")

# Input fields
sepal_length = st.number_input("Sepal Length", value=5.1)
sepal_width = st.number_input("Sepal Width", value=3.5)
petal_length = st.number_input("Petal Length", value=1.4)
petal_width = st.number_input("Petal Width", value=0.2)

# Prediction Button
if st.button("Predict Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    prediction_index = int(prediction.item())
    
    output_species = target_names[prediction_index]
    st.success(f"Result: {output_species} ✨")