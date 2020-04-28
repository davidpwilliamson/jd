
from flask import Flask
from flask import request
from flask import render_template


app = Flask(__name__)

app.config["DEBUG"]  = True

@app.route('/')
def my_form():
    return render_template("form.html")

@app.route('/', methods=['POST'])
def my_form_post():
    number1 = request.form['num1']
    number2 = request.form['num2']
    return "<p>"+number1+" " + number2 + "</p>"
