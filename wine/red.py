from wine import app
from flask import render_template, request

@app.route('/red')
def red():
	return render_template('red.html')