import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from ather_data import good_list,param_rf,param_lr,params_lgb,param_cb,params_xgb,target,categorical_features,\
discrete_feature,continuous_feature,count_columns,TRAIN,model_names
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import pickle


def start_encoding_data():
    global data_encoding_category
    with open('data/data_encoding_category.pkl', 'rb') as file:
        data_encoding_category = pickle.load(file)


def make_enc_target(X):
    global data_encoding_category
    data_encoding_category={}
    X_train = X.copy()
    for feature in categorical_features:
        data_encoding_category[feature] = X_train[X_train[target] == 1].groupby(feature).size() / len(X_train) * 100
    with open('data/data_encoding_category.pkl','wb')  as f:
        pickle.dump(data_encoding_category,f)

def transform_data(X):
    X_train = X.copy()
    d = {}
    for i in X['Years in current job'].value_counts().index:
        if i[:2] == '10':
            d[i] = 10
            continue
        if i[0] == '<':
            d[i] = 0
            continue

    X_train['Years in current job'] = X['Years in current job'].map(d)

    X_train.loc[X['Maximum Open Credit'] > 7000000, 'Maximum Open Credit'] = 7000000

    X_train.fillna(999, inplace=True)

    if __name__ == '__main__':
        make_enc_target(X)
    for i in categorical_features:
        X_train[i] = X_train[i].map(data_encoding_category[i])
        X_train[i].fillna(999,inplace = True)
    if __name__ == '__main__':
        X_train=X_train.drop(target,1)
    X_train_lr = get_data_for_log_reg(X_train)
    return X_train,X_train_lr


def get_data_for_log_reg(X):
    X_train_lr = X.copy()
    for i in categorical_features:
        X_train_lr = pd.concat([X_train_lr, pd.get_dummies(X_train_lr[i], prefix=i)], axis=1)
        X_train_lr.drop(i, 1, inplace=True)

    for i in continuous_feature + discrete_feature:
        X_train_lr[i] = StandardScaler().fit_transform(X_train_lr[[i]])
    new = pd.DataFrame()
    num_columns = X_train_lr.columns.to_list()[:12]
    for col1 in num_columns:
        new[col1 + '_**2'] = (X_train_lr[i]) ** 2
        new[col1 + '_**3'] = (X_train_lr[i]) ** 3
        new[col1 + '_log'] = np.log(X_train_lr[col1] + 2.2)

        for col2 in num_columns:
            new[col1 + '_' + col2 + '_*'] = X_train_lr[col1] * X_train_lr[col2]
            if col1 != col2:
                new[col1 + '_' + col2 + '_/'] = X_train_lr[col1] / X_train_lr[col2]
    X_train_lr = pd.concat([X_train_lr, new[good_list]], 1)
    return X_train_lr


class FitModel():
    def __init__(self, X):
        self.data = X.copy()
        self.y_train = X[target]
        assert self.test_data(),'Структура данных не соответсвует требованиям.'

    def test_data(self):
        columns = self.data.columns.to_list()
        if self.data.shape[1]!=count_columns:
            return False
        for i in categorical_features + continuous_feature + categorical_features:
            if i not in columns:
                return False
            else:
                return True

    def transform_data(self):
        self.X_train,self.X_train_lr = transform_data(self.data)
        # self.X_train_lr = get_data_for_lod_reg(self.X_train)
        self.flag_transform = True

    def fit(self):
        assert self.flag_transform, 'Сначало нужно трансформировать данные'
        for i in model_names:
            x = self.X_train
            if i == 'model_rf':
                model = RandomForestClassifier(**param_rf)
            elif i== 'model_cb':
                model = cb.CatBoostClassifier(**param_cb)
            elif i == 'model_lgb':
                model = lgb.LGBMClassifier(**params_lgb)
            elif i == 'model_xgb':
                model = xgb.XGBClassifier(**params_xgb)
            elif i == 'model_lr':
                model = LogisticRegression(**param_lr)
                x = self.X_train_lr
            model.fit(x, self.y_train)
            with open(f'data/{i}', 'wb') as file:
                pickle.dump(model, file)






if __name__ == '__main__' :
    df = pd.read_csv(TRAIN)
    a = FitModel(df)
    a.transform_data()
    a.fit()
