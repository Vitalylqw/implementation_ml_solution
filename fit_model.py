import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from ather_data import good_list,param_rf,param_lr,params_lgb,param_cb,params_xgb,target,categorical_features,\
discrete_feature,continuous_feature,count_columns
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import pickle


class FitModel():
    def __init__(self, X):
        self.data = X
        # self.target = 'Credit Default'
        # self.categorical_features = ['Home Ownership', 'Tax Liens', 'Purpose', 'Term']
        # self.discrete_feature = ['Years in current job', 'Number of Open Accounts',
        #                          'Years of Credit History', 'Number of Credit Problems',
        #                          'Months since last delinquent', 'Bankruptcies', 'Credit Score']
        # self.continuous_feature = ['Annual Income', 'Maximum Open Credit', 'Current Loan Amount',
        #                            'Current Credit Balance', 'Monthly Debt']
        #
        # self.count_columns = 17

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
        self.X_train = self.data.copy()
        d = {}
        for i in self.data['Years in current job'].value_counts().index:
            if i[:2] == '10':
                d[i] = 10
                continue
            if i[0] == '<':
                d[i] = 0
                continue
        self.X_train['Years in current job'] = self.data['Years in current job'].map(d)

        self.X_train.loc[self.data['Maximum Open Credit'] > 7000000, 'Maximum Open Credit'] = 7000000

        self.X_train.fillna(999, inplace=True)

        for i in categorical_features:
            d = self.enc_target( i )
            self.X_train[i] = self.X_train[i].map(d)
            self.X_train[i] = self.X_train[[i]].fillna(0)

        self.get_data_for_lod_reg()

    def enc_target(self,feature):
        d = self.X_train[self.X_train[target] == 1].groupby(feature).size() / len(self.X_train) * 100
        return d

    def get_data_for_lod_reg(self):
        self.X_train_lr = self.X_train.drop(target, 1)
        for i in categorical_features:
            self.X_train_lr = pd.concat([self.X_train_lr, pd.get_dummies(self.X_train_lr[i], prefix=i)], axis=1)
            self.X_train_lr.drop(i, 1, inplace=True)

        for i in continuous_feature + discrete_feature:
            self.X_train_lr[i] = StandardScaler().fit_transform(self.X_train_lr[[i]])

        new = pd.DataFrame()
        num_columns = self.X_train_lr.columns.to_list()[:12]
        for col1 in num_columns:
            new[col1 + '_**2'] = (self.X_train_lr[i]) ** 2
            new[col1 + '_**3'] = (self.X_train_lr[i]) ** 3
            new[col1 + '_log'] = np.log(self.X_train_lr[col1] + 2.2)

            for col2 in num_columns:
                new[col1 + '_' + col2 + '_*'] = self.X_train_lr[col1] * self.X_train_lr[col2]
                if col1 != col2:
                    new[col1 + '_' + col2 + '_/'] = self.X_train_lr[col1] / self.X_train_lr[col2]

        self.X_train_lr = pd.concat([self.X_train_lr, new[good_list]], 1)
        self.flag_transform = True


    def fit(self):
        assert self.flag_transform, 'Сначало нужно трансформировать данные'
        model_rf = RandomForestClassifier(**param_rf)
        model_rf.fit(self.X_train,self.y_train)
        with open('data/model_rf', 'wb') as file:
            pickle.dump(model_rf, file)

        model_cb = cb.CatBoostClassifier(**param_cb)
        model_cb.fit(self.X_train, self.y_train)
        with open('data/model_cb', 'wb') as file:
            pickle.dump(model_cb, file)

        model_lgb = lgb.LGBMClassifier(**params_lgb)
        model_lgb.fit(self.X_train, self.y_train)
        with open('data/model_lgb', 'wb') as file:
            pickle.dump(model_lgb, file)

        model_xgb = xgb.XGBClassifier(**params_xgb)
        model_xgb.fit(self.X_train, self.y_train)
        with open('data/model_xgb', 'wb') as file:
            pickle.dump(model_xgb, file)

        model_lr = LogisticRegression(**param_lr)
        model_lr.fit(self.X_train, self.y_train)
        with open('data/model_lr', 'wb') as file:
            pickle.dump(model_lr, file)


if __name__ == '__main__' :
    df = pd.read_csv('data/train.csv')
    a = FitModel(df)
    a.transform_data()
    a.fit()
