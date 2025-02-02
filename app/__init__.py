from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from .routes import routes  # Import the blueprint

db = SQLAlchemy()

def create_app():
    print("Creating Flask app instance...")  # Debug print
    app = Flask(__name__)
    app.config.from_object('config.Config')
    db.init_app(app)
    
    # Register the blueprint for routes
    app.register_blueprint(routes)
    print("Blueprint registered.")
    return app
