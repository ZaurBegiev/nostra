import os
import logging

from flask import Flask, render_template

from config import Config


logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config.from_object(Config)


@app.route('/')
def hello_world():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
