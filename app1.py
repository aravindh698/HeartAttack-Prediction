#import libraries
import numpy as np
from flask import Flask, request, render_template
import pickle

#Initialize the flask App
app = Flask(__name__,template_folder='template')
f = open('model.pkl', 'rb')
model1=pickle.load(f)


#default page of our web-app
@app.route('/')
def home():
    return render_template('index1.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model1.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index1.html', prediction_text='Heart Attack Chances are :{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)