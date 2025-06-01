from flask import Blueprint, flash, render_template, request, redirect, url_for
from datetime import datetime
from app import db
from models.patient import Patient
from routes.auth import login_required
from services.segmentation import run_segmentation_pipeline
from models.history import PredictionHistory
import os, uuid
from werkzeug.utils import secure_filename

patients_bp = Blueprint('patients', __name__)

@patients_bp.route('/patients')
@login_required
def list_patients():
    patients = Patient.query.order_by(Patient.id.desc()).all()
    return render_template('patients.html', patients=patients)

@patients_bp.route('/patients/add', methods=['POST'])
@login_required
def add_patient():
    name = request.form['name']
    dob = datetime.strptime(request.form['dob'], '%Y-%m-%d').date()
    gender = request.form['gender']
    address = request.form['address']
    phone = request.form['phone']
    last_visit = request.form.get('last_visit')
    last_diagnosis = request.form.get('last_diagnosis')
    status = request.form.get('status', 'Chưa khám')
    
    new_patient = Patient(
        name=name,
        dob=dob,
        gender=gender,
        address=address,
        phone=phone,
        last_visit=datetime.strptime(last_visit, '%Y-%m-%d') if last_visit else None,
        last_diagnosis=last_diagnosis,
        status=status          
    )
    
    db.session.add(new_patient)
    db.session.commit()
    return redirect(url_for('patients.list_patients'))


@patients_bp.route('/patients/<int:patient_id>')
@login_required
def patient_detail(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    histories = db.session.query(PredictionHistory).filter(PredictionHistory.patient_id == patient_id).order_by(PredictionHistory.created_at.desc()).all()
    return render_template("patient_detail.html", patient=patient, histories=histories)

@patients_bp.route('/patients/<int:patient_id>/segment', methods=['POST'])
@login_required
def segment_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)

    # Nhận file
    file = request.files['image']
    if not file:
        flash("Không tìm thấy file ảnh.", "danger")
        return redirect(url_for('patients.patient_detail', patient_id=patient_id))

    # Tạo file_id mới
    file_id = str(uuid.uuid4())[:8]
    filename = f"patient_{patient.id}_{file_id}.jpg"  # Đặt tên mới không lấy tên gốc
    upload_path = os.path.join('static/uploads', filename)
    file.save(upload_path)

    # Chạy pipeline
    result = run_segmentation_pipeline(upload_path, file_id=f"patient_{patient.id}_{file_id}")

    # Ghi vào DB
    history = PredictionHistory(
        patient_id=patient.id,
        image_name=filename,
        mask_path=result['mask_path'],
        overlay_path=result.get('overlay_path'),
        bbox_path=result.get('bbox_path'),
        heatmap_path=result.get('heatmap'),
        overlay_alpha_path=result.get('overlay_alpha'),
        bbox_alpha_path=result.get('bbox_alpha'),
        heatmap_alpha_path=result.get('heatmap_alpha'),
        accuracy=result.get('accuracy', 0)
    )
    db.session.add(history)

    # Cập nhật thông tin bệnh nhân
    patient.last_visit = datetime.utcnow()
    patient.last_diagnosis = result.get('diagnosis', 'Chưa xác định')
    patient.status = 'Đã khám'

    db.session.commit()
    flash("Ảnh đã được phân tích và lưu thành công!", "success")
    
    return redirect(url_for('patients.patient_detail', patient_id=patient_id))
