from flask import Blueprint, flash, render_template, request, jsonify
from models.history import PredictionHistory
from extensions import db
import json

from routes.auth import login_required

labeling_bp = Blueprint("labeling", __name__)


@labeling_bp.route("/labeling")
@login_required
def labeling_list():
    histories = PredictionHistory.query.all()
    return render_template("labeling_lists.html", histories=histories)


@labeling_bp.route("/labeling/<int:history_id>", methods=["GET"])
@login_required
def labeling_page(history_id):
    history = PredictionHistory.query.get_or_404(history_id)
    return render_template("labeling_tool.html", history=history)

@labeling_bp.route("/labeling/<int:history_id>/save", methods=["POST"])
@login_required
def save_labels(history_id):
    history = PredictionHistory.query.get_or_404(history_id)
    label_data = request.json.get("labels")
    
    # Lưu kết quả chỉnh sửa vào trường riêng hoặc cập nhật lại mask_path nếu cần
    history.manual_labels = json.dumps(label_data)
    history.labeled = True
    db.session.commit()

    return jsonify({"status": "success", "message": "Lưu thành công"})


@labeling_bp.route("/labeling/<int:history_id>/view", methods=["GET"])
@login_required
def view_labeled_image(history_id):
    history = PredictionHistory.query.get_or_404(history_id)
    labels = None
    if history.manual_labels:
        try:
            labels = json.loads(history.manual_labels)
        except Exception:
            flash("Không thể đọc nhãn BBox", "danger")

    if not labels:
        flash("Chưa có nhãn nào được lưu", "warning")

    return render_template(
        "view_labels.html",
        history=history,
        labels=labels
    )