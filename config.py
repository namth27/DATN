import os

class Config:
    SECRET_KEY = ""

    SQLALCHEMY_DATABASE_URI = 'sqlite:///dental_ai.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = 'static/uploads'
    RESULT_FOLDER = 'static/results'

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
