# coding=utf-8

from flask import Flask,request
from datetime import datetime
from flask_cors import CORS
from get_ifo import predict_one,start_models
from fit_model import  start_encoding_data

import json


start_models()
start_encoding_data()

app = Flask(__name__)
ser_name = "API borrower quality assessment"
CORS(app)

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
    return response


@app.route("/")
def hello():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f"Welcome to {ser_name}  <br> {timestamp}"


@app.route("/predict_one/",methods=['GET','POST'])
def predict_one_element():
    """Возвращает предсказание по одной записи"""
    error = []
    dic = {}
    try:
        data = request.values
    except Exception as E:
        dic['Тип'] = 'can not get parameters request'
        dic['exeption'] = str(E)
        error.append(dic)
        res = {'status':False,'Error':error}
        print('Не удалось считать параметры')
        return json.dumps(res)
    try:
        res = predict_one(data)
    except Exception as E:
        dic['Тип'] = 'can not get predict'
        dic['exeption'] = str(E)
        error.append(dic)
        res = {'status':False,'Error':error}
        print("Не удалось выполнить предикт")
        return json.dumps(res)

    return json.dumps(res)




app.run()