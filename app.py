import os
import logging

from flask import Flask, render_template, request, Response, flash, jsonify
import pandas as pd
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = os.environ['FLASK_KEY_NOSTRA']
app.config['UPLOAD_FOLDER'] = 'C:\\Users\\Zaur\\nostra\\uploads'

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'}
OK = Response(status=200)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/upload_dataset', methods=['GET', 'POST'])
def upload_dataset():
    if request.method == 'POST':

        if 'inputFile' not in request.files:
            flash('No file part')
            return OK
        file = request.files['inputFile']

        if file.filename == '':
            flash('No selected file')
            return OK

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            preview = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename)).head()
            table_view = preview.to_json()
            return jsonify(table_view)
        else:
            return OK


if __name__ == '__main__':
    app.run()
