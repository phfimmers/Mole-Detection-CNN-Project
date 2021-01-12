import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='./template')
app.config['EXPLAIN_TEMPLATE_LOADING'] = True
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif', '.bmp']
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # integrate prediction here
        # integrate reply here
     
    return redirect(url_for('index'))

if __name__ == "__main__":
    # You want to put the value of the env variable PORT if it exist
    # (some services only open specifiques ports)
    port = int(os.environ.get('PORT', 5000))
    # Threaded option to enable multiple instances for
    # multiple user access support
    # You will also define the host to "0.0.0.0" because localhost
    # will only be reachable from inside de server.
    app.run(host="0.0.0.0", threaded=True, port=port)