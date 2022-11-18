# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 19:53:40 2022

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:46:32 2022

@author: DELL
"""

from flask import Flask, render_template, request,Response
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
app = Flask(__name__)
import cv2
from gtts import gTTS



indexs=['a','b','c','d','e','f','g','h','i']

model = load_model('sign.h5')
camera=cv2.VideoCapture(0)
model.make_predict_function()


def predict_label(img_path):
        i = tf.keras.utils.load_img(img_path, target_size=(64,64))
        i = img_to_array(i)/255.0
        i=np.expand_dims(i,axis=0)
        p = model.predict(i)
        return np.argmax(p)

def generate_frames():
    while True:
        ret,frame =camera.read()
        copy = frame.copy()
        copy = copy[150:150+200,50:50+200]
        cv2.imwrite('image.jpg',copy)
        copy_img = tf.keras.utils.load_img('image.jpg', target_size=(64,64))
        x = img_to_array(copy_img)
        x = np.expand_dims(x, axis=0)
        pred = np.argmax(model.predict(x), axis=1)
        y = pred[0]
        l=str(indexs[y])
        audio =gTTS(text=l, lang="en", slow=False)
        audio.save("example.wav")
        cv2.putText(frame,'The Predicted Alphabet is: '+str(indexs[y]),(150,150,),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        ret,jpg = cv2.imencode('.jpg', frame)
        frame=jpg.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate():

        with open("example.wav", "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)   
    
# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("indexsd.html")

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
    return render_template("indexsd.html", prediction = p, img_path = img_path)

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route("/audio")
def audio():
    return Response(generate(), mimetype="audio/x-wav")


if __name__ =='__main__':
	#app.debug = True
    app.run(debug=True,port=9985,use_reloader=False)