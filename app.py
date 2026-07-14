# app.py
from flask import Flask, request, render_template
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
# Load the built-in Scikit-Learn model
model_path = 'model.pkl'
try:
    model_stuff = joblib.load(model_path)
    model = model_stuff['model']
    target_names = model_stuff['target_names']
except FileNotFoundError:
    print("Error: model.pkl not found. Run train.py first!")
    model = None
    target_names = []

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: Model file not loaded.")
        
    try:
        # 1. Extract values from form
        float_features = [float(x) for x in request.form.values()]
        
        # 2. Reshape for Scikit-Learn's interface
        final_features = np.array([float_features])
        
        # 3. Predict (Scikit-Learn natively converts this safely)
        prediction_index = model.predict(final_features)[0]
        output_species = target_names[prediction_index]

        return render_template(
            'index.html', 
            prediction_text=f'Predicted Species: {output_species.capitalize()}'
        )
        
    except ValueError:
        return render_template(
            'index.html', 
            prediction_text='Error: Please enter valid numbers in all fields.'
        )
    except Exception as e:
        return render_template(
            'index.html', 
            prediction_text=f'Prediction Error: {str(e)}'
        )

if __name__ == "__main__":
    app.run(debug=True)