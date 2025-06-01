from functools import wraps
from flask import Blueprint, Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename
from app import db
from models.patient import Patient
import os, uuid
from datetime import datetime
from models.prediction import Prediction
from models.history import PredictionHistory
from models.user import User
from routes.auth import login_required

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
@login_required
def index():
    total_patients = Patient.query.count()
    total_predictions = Prediction.query.count() + PredictionHistory.query.count()

    # Lấy bệnh nhân gần đây
    patients_recent = Patient.query.order_by(Patient.last_visit.desc().nullslast()).limit(5).all()

    # Tính highest_accuracy cho mỗi bệnh nhân
    for patient in patients_recent:
        histories = PredictionHistory.query.filter_by(patient_id=patient.id).all()
        if histories:
            patient.highest_accuracy = max((h.accuracy or 0) for h in histories)
        else:
            patient.highest_accuracy = None

    # Lấy phân tích gần đây
    predictions_recent = (
        db.session.query(
            PredictionHistory,
            Patient.name.label('patient_name'),
            Patient.last_diagnosis.label('diagnosis')
        )
        .join(Patient, PredictionHistory.patient_id == Patient.id)
        .order_by(PredictionHistory.created_at.desc())
        .limit(6)
        .all()
    )

    predictions_recent = [
        {
            "patient_id": ph.patient_id,
            "patient_name": name,
            "created_at": ph.created_at,
            "overlay_path": ph.overlay_path,
            "diagnosis": diagnosis
        }
        for ph, name, diagnosis in predictions_recent
    ]

    # Tính accuracy trung bình toàn hệ thống
    all_histories = PredictionHistory.query.filter(PredictionHistory.accuracy.isnot(None)).all()
    if all_histories:
        average_accuracy = round(sum(h.accuracy for h in all_histories) / len(all_histories), 2)
    else:
        average_accuracy = 0

    return render_template(
        'index.html',
        total_patients=total_patients,
        total_predictions=total_predictions,
        patients_recent=patients_recent,
        predictions_recent=predictions_recent,
        average_accuracy=average_accuracy
    )
