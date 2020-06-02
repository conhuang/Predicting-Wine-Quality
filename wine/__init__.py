from flask import Flask
from flask_wtf.csrf import CSRFProtect
# from config import Config

app = Flask(__name__)
csrf = CSRFProtect(app)
csrf.init_app(app)

# app.config.from_object(Config)



from wine import home, white, red