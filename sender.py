# coding=utf-8
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import  classification_report

real_ansver = pd.read_csv('data/y_test.csv')
test_df = pd.read_csv('data/test.csv')
url = 'http://127.0.0.1:5000/predict_one/'
def test_many():
    n = 100
    predict =[]
    test = real_ansver.iloc[:n]
    for i in range(n):
        ask_data = test_df.drop('Credit Default',axis=1).iloc[i,:].to_dict()
        data = ask_data
        response = requests.post(url,data = data).json()
        predict.append(int(response['answer']))
        # print(int(response['answer']))
    print(classification_report(test,np.array(predict)))


def test_one():
    ask_data = test_df.drop('Credit Default', axis=1).iloc[5, :].to_dict()
    response = requests.post(url, data=ask_data).json()
    print(response)


test_one()