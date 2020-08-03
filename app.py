import os
import logging

from flask import Flask, render_template, Response


logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = os.environ['FLASK_KEY_NOSTRA']

OK = Response(status=200)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/tabuator')
def tabulator():
    return render_template('tabulator.html')


if __name__ == '__main__':
    app.run()
