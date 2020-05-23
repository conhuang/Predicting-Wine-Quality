from flask import Flask

app = Flask(__name__)

from wine import home,white, red