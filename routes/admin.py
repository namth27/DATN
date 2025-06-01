import io
import json
import os
import zipfile
from flask import Blueprint, current_app, flash, render_template, request, redirect, send_file, url_for
from app import db
from models.history import PredictionHistory
from models.user import User
from datetime import datetime
from PIL import Image, ImageDraw
from routes.auth import admin_required

admin_bp = Blueprint('admin', __name__)

@admin_bp.route("/admin")
def admin_dashboard():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template("admin_dashboard.html", users=users)

@admin_bp.route("/admin/add_user", methods=["POST"])
@admin_required
def add_user():
    username = request.form["username"]
    email = request.form["email"]
    password = request.form["password"]
    role = request.form["role"]

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        flash("Email đã tồn tại", "warning")
        return redirect(url_for("admin.admin_dashboard") + "#addUserModal")  # Gợi ý: quay lại đúng trang Dashboard

    user = User(username=username, email=email, role=role)
    user.set_password(password)

    db.session.add(user)
    db.session.commit()
    flash("Tạo tài khoản thành công", "success")
    return redirect(url_for("admin.admin_dashboard"))



@admin_bp.route("/admin/user/<int:user_id>")
@admin_required
def view_user(user_id):
    user = User.query.get_or_404(user_id)
    return render_template("admin_user_detail.html", user=user)

@admin_bp.route("/admin/user/<int:user_id>/edit", methods=["GET", "POST"])
@admin_required
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    if request.method == "POST":
        user.username = request.form['username']
        user.role = request.form['role']
        user.is_active = 'is_active' in request.form
        db.session.commit()
        return redirect(url_for('admin.admin_dashboard'))
    return render_template("admin_edit_user.html", user=user)

@admin_bp.route("/admin/user/<int:user_id>/deactivate")
@admin_required
def deactivate_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.is_active:
        user.is_active = False
        db.session.commit()
        flash("User has been deactivated successfully.", "success")
    else:
        flash("User is already deactivated.", "warning")
    return redirect(url_for('admin.admin_dashboard'))

@admin_bp.route("/admin/user/<int:user_id>/activate")
@admin_required
def activate_user(user_id):
    user = User.query.get_or_404(user_id)
    if not user.is_active:
        user.is_active = True
        db.session.commit()
        flash("User has been activated successfully.", "success")
    else:
        flash("User is already active.", "warning")
    return redirect(url_for('admin.admin_dashboard'))

@admin_bp.route("/images")
@admin_required
def admin_images():
    bbox_filter = request.args.get("bbox")  # 'yes' hoặc 'no'
    seg_filter = request.args.get("seg")    # 'yes' hoặc 'no'

    query = PredictionHistory.query

    if bbox_filter == "yes":
        query = query.filter(PredictionHistory.manual_labels != None)
    elif bbox_filter == "no":
        query = query.filter(PredictionHistory.manual_labels == None)

    if seg_filter == "yes":
        query = query.filter(PredictionHistory.manual_segmentation != None)
    elif seg_filter == "no":
        query = query.filter(PredictionHistory.manual_segmentation == None)

    histories = query.order_by(PredictionHistory.created_at.desc()).all()
    return render_template("admin_images.html", histories=histories)


@admin_bp.route("/images/export_bbox_txt", methods=["POST"])
@admin_required
def export_bbox_txt():
    selected_ids = request.form.get("selected_ids")
    try:
        ids = json.loads(selected_ids)
    except:
        flash("Không thể đọc danh sách ảnh được chọn.", "danger")
        return redirect("/images")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        histories = PredictionHistory.query.filter(PredictionHistory.id.in_(ids)).all()
        for h in histories:
            if not h.manual_labels:
                continue
            try:
                labels = json.loads(h.manual_labels)
                txt_data = ""
                for box in labels:
                    x, y = box["x"], box["y"]
                    w, h_ = box["width"], box["height"]
                    cls = box["name"]
                    txt_data += f"{cls} {x} {y} {w} {h_}\n"
                filename = f"{h.image_name.rsplit('.', 1)[0]}.txt"
                zf.writestr(filename, txt_data)
            except Exception as e:
                print("Lỗi khi xử lý bbox:", e)

    buffer.seek(0)
    return send_file(buffer, mimetype='application/zip', as_attachment=True, download_name="bbox_labels.zip")


@admin_bp.route("/images/export_segment_mask", methods=["POST"])
@admin_required
def export_segment_mask():
    selected_ids = request.form.get("selected_ids")
    try:
        ids = json.loads(selected_ids)
    except:
        flash("Không thể đọc danh sách ảnh được chọn.", "danger")
        return redirect("/images")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        histories = PredictionHistory.query.filter(PredictionHistory.id.in_(ids)).all()
        for h in histories:
            if not h.manual_segmentation:
                continue
            try:
                polygons = json.loads(h.manual_segmentation)
                image_path = os.path.join(current_app.static_folder, "uploads", h.image_name)
                with Image.open(image_path) as img:
                    mask = Image.new("L", img.size, 0)
                    draw = ImageDraw.Draw(mask)
                    for segment in polygons:
                        points = [(p["x"], p["y"]) for p in segment]
                        draw.polygon(points, fill=255)
                    mask_filename = f"{h.image_name.rsplit('.', 1)[0]}_mask.png"
                    img_byte_arr = io.BytesIO()
                    mask.save(img_byte_arr, format='PNG')
                    zf.writestr(mask_filename, img_byte_arr.getvalue())
            except Exception as e:
                print("Lỗi khi xử lý segmentation:", e)

    buffer.seek(0)
    return send_file(buffer, mimetype='application/zip', as_attachment=True, download_name="segmentation_masks.zip")

