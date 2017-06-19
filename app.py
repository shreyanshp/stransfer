#our web app framework!

#you could also generate a skeleton from scratch via
from flask import Flask, render_template,request, current_app, jsonify
import io
import base64
import re
import os
from os.path import exists
from sys import stdout
import strans as st

#initalize our flask app
app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	data = {}
	try:
		data = request.get_data()
		imgstr = re.search(r'base64,(.*)',data).group(1)
		data = imgstr
	except KeyError:
		return jsonify(status_code='400', msg='Bad Request'), 400

	data = base64.b64decode(data)
	image = io.BytesIO(data)
	mynet = "rain-princess-network"
	mytrans = st.strans(image, mynet)
	return mytrans

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run('0.0.0.0', port)
