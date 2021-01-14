import os
import sys

import cv2
from flask import Flask, render_template, request, redirect, url_for, abort
import numpy as np
from werkzeug.utils import secure_filename
from keras.models import load_model
import requests

from preprocessing.preprocess import selma_secret_sauce

# download model
url = 'https://b0ykepubbucket.s3-eu-west-1.amazonaws.com/64model.h5'
r = requests.get(url, stream = True)
chunk_progress = 0
with open("modelfile.h5", "wb") as modelfile:
    for chunk in r.iter_content(chunk_size = 8388608):
        if chunk:
            modelfile.write(chunk)
            chunk_progress += 1
            print(f"Downloaded: {chunk_progress*8}MB\n")
            sys.stdout.flush()

# load the model
model = load_model('modelfile.h5')

# start the app
app = Flask(__name__, template_folder='./template')
app.config['EXPLAIN_TEMPLATE_LOADING'] = True
app.config['MAX_CONTENT_LENGTH'] = 2500 * 2500
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
app.config['UPLOAD_PATH'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext.lower() not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        # integrate preprocessing here
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        image = selma_secret_sauce(image, advanced = False, for_training = False, model_size = (64,64))
        # integrate prediction here
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        print(f"Predicted: {prediction[0][0]}\n")
        sys.stdout.flush()
        # integrate reply here
        if prediction[0][0] >= 0.5:
            print("redirecting to doctor")
            sys.stdout.flush()
            return "Please have it checked by a doctor."
        else:
            print("redirecting to no doctor")
            sys.stdout.flush()
            return "This looks harmless"
    
    return redirect(url_for('upload_files'))

if __name__ == "__main__":
    # You want to put the value of the env variable PORT if it exist
    # (some services only open specifiques ports)
    port = int(os.environ.get('PORT', 5000))
    # Threaded option to enable multiple instances for
    # multiple user access support
    # You will also define the host to "0.0.0.0" because localhost
    # will only be reachable from inside de server.
    app.run(host="0.0.0.0", threaded=True, port=port)
