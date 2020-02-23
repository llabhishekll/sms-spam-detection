from flask import Flask, render_template, request, session
from model.process import Model

app = Flask(__name__)

@app.route('/')
def index():
	return(render_template('index.html'))

data = str()

@app.route('/result', methods=["POST","GET"])
def result():
	data = request.form.get('message')
	result = str(Model(data)).strip('[\'\']')
	return(render_template('result.html',result=result))

if __name__ == '__main__':
	app.run()