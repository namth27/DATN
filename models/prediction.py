from app import db
from datetime import datetime

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'))
    overlay_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.now)