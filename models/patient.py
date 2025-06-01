from app import db
from datetime import datetime

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    address = db.Column(db.String(255))
    phone = db.Column(db.String(20)) 
    last_visit = db.Column(db.Date, nullable=True)
    last_diagnosis = db.Column(db.String(255), nullable=True)
    status = db.Column(db.String(50))
