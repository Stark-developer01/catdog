from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

from keras.models import model_from_json
import numpy as np

with open('catdog.json', 'r') as json_file:
    json_savedModel= json_file.read()
model_j = tf.keras.models.model_from_json(json_savedModel)
model_j.load_weights('model\catdog_weights.h5')

def pred(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    #cv2_imshow(img)
    img = np.reshape(img, (1, 256, 256, 3))
    pred = model_j.predict(img)
    #pred = [[1, 0]]
    if pred[0][0]>pred[0][1]:
        return 'cat'
    else:
        return 'dog'

app  = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    #return 'f'
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = pred(file_path)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(preds)               # Convert to string
        return result
    return None

if __name__=="__main__":
    app.run(debug=True)
