import os


class Config:
    DEBUG = True
    SECRET_KEY = os.environ['FLASK_KEY_NOSTRA']