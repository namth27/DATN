from flask import Flask
from config import Config
from extensions import db

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)

    from routes.main import main_bp
    from routes.history import history_bp
    from routes.patients import patients_bp
    from routes.auth import auth_bp
    from routes.admin import admin_bp
    from routes.labeling import labeling_bp
    from routes.segment import segment_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(patients_bp)
    app.register_blueprint(labeling_bp)
    app.register_blueprint(segment_bp)
    return app

app = create_app()
