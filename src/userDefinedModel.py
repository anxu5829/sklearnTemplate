import numpy as np
# import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

class UserLightgbmModel(BaseEstimator,TransformerMixin):

    def __init__(self):
        self.data = None
        self.model = None
        self.modelparams = {
                'boosting_type' : 'gbdt',
                'num_leaves' : 48,
                'max_depth' : -1,
                'learning_rate' : 0.05,
                'n_estimators' : 252,
                'max_bin' : 425,
                'subsample_for_bin': 50000,
                'objective' : 'xentropy',
                'min_split_gain' : 0,
                 'min_child_weigt' : 5,
                 'min_child_sample'  : 10,
                 'subsample' : 1,
                 'subsample_freq' : 1,
                 'colsample_bytree' : 1,
                 'reg_alpha' :3,
                 'reg_lambda' : 5,
                 'seed' : 1000,
                  'silent' : True,
                #   'early_stopping_rounds':20,
                #   'metric' : 'logloss'
             }
        self.train_data  = None
        self.train_label = None 
        self.test_data   = None
        self.test_label  = None 

        # list of cols name you may use to build lightGBM
        self.feature_name = None
        self.categorical_feature = None

    def _get_train_and_val(self):

        
        self.data.train_val_split_by_time(grain = 'day')
        self.data.getSplitedData()
        self.train_data,self.train_label,self.test_data,self.test_label = self.data.getSplitedData()

    def _get_column_used(self):
        self.data.setUselessVar([])
        # self.data.useBestVariable()
        self.categorical_feature = list(
            set(self.data.data.columns) &
             set(self.data.categoryCols)
             )
        
        self.feature_name =  self.data.data.columns
                
        
    

        # self.data.data = self.data.data.loc[:,self.feature_name]


    def _encodeCategoricalCols(self):
        for cols in self.categorical_feature :
            if cols != 'day':
                col_encoder = LabelEncoder()
                col_encoder.fit(self.data.data[cols]) 
                self.data.data[cols] = col_encoder.transform(self.data.data[cols])

    def fit(self,X,y = None):
        # X is a instance of Data() Class
        self.data = X
        self._get_column_used()
        # self._encodeCategoricalCols()
        self._get_train_and_val()

        # for other purpose , use lgb like training process
        trainSet = lgb.Dataset(
                self.train_data,
                label = self.train_label,
                categorical_feature=self.categorical_feature
            )
        testSet  = lgb.Dataset(
                self.test_data,
                label = self.test_label,
                categorical_feature=self.categorical_feature
            )

        gbm = lgb.train(self.modelparams,
                trainSet,
                valid_sets=[trainSet,testSet]  # eval training data
                )




        # self.model = lgb.LGBMClassifier(**self.modelparams)
        # self.model.fit(self.train_data,self.train_label,
        #                 eval_set=[(self.train_data,self.train_label),(self.test_data,self.test_label)],eval_metric=['logloss','auc'],
        #                 ,early_stopping_rounds= 20,
        #                 categorical_feature=self.categorical_feature
        #             )
        return self
        

    def predict(self,X,y = None):
        return X

    def score(self,X,y = None):
        return(sum(self.predict(X)))

    def evaluate(self,X,y):
        return 0

        