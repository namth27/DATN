from app import app, db
from models.patient import Patient
from models.history import PredictionHistory
from models.prediction import Prediction

with app.app_context():
    db.drop_all()
    db.create_all()
    print("Database đã tạo lại thành công.")