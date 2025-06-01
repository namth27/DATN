from flask import Blueprint, flash, render_template, request, jsonify
from models.history import PredictionHistory
from extensions import db
import json

from routes.auth import login_required

segment_bp = Blueprint("segment", __name__)


@segment_bp.route("/segment")
@login_required
def segment_list():
    histories = PredictionHistory.query.all()
    return render_template("segment_lists.html", histories=histories)


@segment_bp.route("/segment/<int:history_id>", methods=["GET"])
@login_required
def segment_page(history_id):
    history = PredictionHistory.query.get_or_404(history_id)
    return render_template("segment_tool.html", history=history)


@segment_bp.route("/segment/<int:history_id>/save", methods=["POST"])
@login_required
def save_segmentation(history_id):
    history = PredictionHistory.query.get_or_404(history_id)
    segmentation_data = request.json.get("polygons")

    if segmentation_data:
        try:
            # Kiểm tra dữ liệu trước khi lưu vào cơ sở dữ liệu
            json.dumps(segmentation_data)
            history.manual_segmentation = json.dumps(segmentation_data)
            history.segmented = True
            db.session.commit()
            return jsonify({"status": "success", "message": "Lưu segmentation thành công"})
        except TypeError as e:
            print(f"Error saving segmentation: {e}")
            return jsonify({"status": "error", "message": "Dữ liệu segmentation không hợp lệ"}), 400
    else:
        return jsonify({"status": "error", "message": "Dữ liệu segmentation không có"}), 400


@segment_bp.route("/segment/<int:history_id>/view", methods=["GET"])
@login_required
def view_segmented_image(history_id):
    history = PredictionHistory.query.get_or_404(history_id)
    
    polygons = None
    if history.manual_segmentation:
        try:
            polygons = json.loads(history.manual_segmentation)
        except (json.JSONDecodeError, TypeError):
            flash("Không thể đọc nhãn segmentation", "danger")
            polygons = []  # Đảm bảo polygons có giá trị mặc định là danh sách rỗng nếu không thể load

    if not polygons:
        flash("Chưa có nhãn segmentation nào được lưu", "warning")
    
    return render_template(
        "view_segmentation.html",
        history=history,
        segments=polygons
    )