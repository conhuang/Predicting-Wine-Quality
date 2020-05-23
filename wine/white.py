from wine import app
from flask import render_template, request

@app.route('/white')
def white():
	return render_template('white.html')