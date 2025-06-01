from app import db
from datetime import datetime

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    image_name = db.Column(db.String(255), nullable=False)

    mask_path = db.Column(db.String(255))
    overlay_path = db.Column(db.String(255))
    overlay_alpha_path = db.Column(db.String(255))
    bbox_path = db.Column(db.String(255))
    bbox_alpha_path = db.Column(db.String(255))
    heatmap_path = db.Column(db.String(255))     
    heatmap_alpha_path = db.Column(db.String(255))   
    
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    manual_labels = db.Column(db.Text, nullable=True)
    labeled = db.Column(db.Boolean, default=False)
    
    manual_segmentation = db.Column(db.Text, nullable=True)
    segmented = db.Column(db.Boolean, default=False)
    accuracy = db.Column(db.Float)
