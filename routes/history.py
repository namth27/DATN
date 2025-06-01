from flask import Blueprint, render_template
from app import db
from models.patient import Patient
from models.history import PredictionHistory

history_bp = Blueprint('history', __name__)

@history_bp.route('/history')
def history():
    # Lấy danh sách lịch sử phân tích từ mới đến cũ
    histories = db.session.query(PredictionHistory, Patient).join(Patient).order_by(PredictionHistory.created_at.desc()).all()

    return render_template('history.html', histories=histories)

