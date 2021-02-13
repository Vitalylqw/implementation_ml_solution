# coding=utf-8
from ather_data import data_columns, model_prediction, split,model_names,discrete_feature,continuous_feature
from fit_model import transform_data
import pickle

import pandas as pd
import numpy as np





def start_models():
    global models
    models= {}
    for i in model_names:
        with open(f'data/{i}','rb') as file:
            models[i] = pickle.load(file)


def get_predictions(data, data_lr):
    preds = pd.DataFrame()
    for model in model_prediction[0]:
        x = data
        if model == 'model_lr':
            x = data_lr
        preds[model] = models[model].predict_proba(x)[:, 1]
        comand = 'preds' + model_prediction[1]
        pred = eval(comand)
        return np.where(pred > split, 1, 0)


def get_data(data, param):
    error = []
    dic = {}
    try:
        insert_data = [data[i] for i in param]
    except Exception as E:
        print('Не удалось извлечь данные')
        dic['type'] = 'can not get parameters request'
        dic['exeption'] = str(E)
        error.append(dic)
        return None, {'status': False, 'Error': error}
    else:
        return insert_data, None


def predict_one(data):
    # Получим переданный параметр
    error = []
    dic = {}
    insert_data, answer = get_data(data, data_columns)
    if not insert_data:
        return answer
    df = pd.DataFrame(columns=data_columns)
    df.loc[0]=insert_data
    float_feature = discrete_feature+continuous_feature
    float_feature.remove('Years in current job')
    df[float_feature]= df[float_feature].astype(float)
    try:
        x, x_lr = transform_data(df)
    except Exception as E:
        print('Не удалось трансофрмировать данные')
        dic['type'] = 'can not transform data'
        dic['exeption'] = str(E)
        error.append(dic)
        return {'status': False, 'Error': error}
    try:
        result = get_predictions(x, x_lr)
    except Exception as E:
        print('Не удалось получить ответ от модели')
        dic['type'] = 'can not to get response from model'
        dic['exeption'] = str(E)
        error.append(dic)
        return {'status': False, 'Error': error}

    return {'status': True, 'answer': str(result[0])}
