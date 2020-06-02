from wine import app
from flask import render_template, request

@app.route('/theory')
def red():
	return render_template('red.html')