from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
app = Flask(__name__)

indexs=['a','b','c','d','e','f','g','h','i']

model = load_model('sign.h5')

model.make_predict_function()

def predict_label(img_path):
        i = tf.keras.utils.load_img(img_path, target_size=(64,64))
        i = img_to_array(i)/255.0
        i=np.expand_dims(i,axis=0)
        p = model.predict(i)
        return np.argmax(p)


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path ="static/"+img.filename	
        img.save(img_path)
        p = predict_label(img_path)
        p=indexs[p]
    return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
    app.run(debug=True,port=9989,use_reloader=False)