# coding=utf-8
import sys
import os
import re
from pathlib import Path



# Import fast.ai Library
from fastai import *
from fastai.vision import *
from werkzeug.utils import secure_filename


# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define a flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# import model and load the learner
path = Path('/Users/ltaing/Documents/FASTAI/app/')
learn = load_learner(path=path, file='export.pkl')
classes = learn.data.classes
defaults.device = torch.device('cpu')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


# route for prediction
@app.route('/predict', methods=['POST'])
def upload_file():
    # Get the file from post request
    f = request.files['file']

    if f and allowed_file(f.filename):

        # Save the file to ./uploads
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        url_file = url_for('static', filename=f'uploads/{filename}')
        img_obj = open_image(path/'static'/'uploads'/filename)

        # Make prediction
        pred_class, pred_idx, outputs = learn.predict(img_obj)
        return redirect(url_for('static', response=pred_class.obj, filename=url_file))

if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0", debug=True)