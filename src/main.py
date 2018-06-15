import config
import os
import easydict as edict
from dataLoader import DataLoader,Data
from userDefinedModel import UserLightgbmModel
import logging
import argparse
# from sklearn.grid_search import GridSearchCV


def _reading_data():
    print(config.USER)

    # step2 the way to load_data
    # load data contains :
        # the way to load data
        # the way to preprocess with data
        # doing some special data cleaning process
    trainFilepath    = os.path.join(os.getcwd(),"data",config.FILENAME)
    trainDataLoader  = DataLoader(trainFilepath)
    train_data       = trainDataLoader.load_data(useSpark= False,interactive=False)

    train_data.save_data(os.getcwd())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload_data", dest="reload", type=bool, default=False
    , help="should we reload data or not")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    logging.basicConfig(level = logging.NOTSET)
    args = parse_args()

    # step1 the way to use config file
    if args.reload :
        _reading_data()    


    train = Data()
    train = train.load_data(os.getcwd())


    model = UserLightgbmModel()


    model.fit(train)
    # # # get split of trainset and val set
    # train.getVariable()

    # train.setUseLessVar([])
    # train.train_eval_split_by_time(grain = 'day')


    # train_data,train_label,test_data,test_label = train_data.getSplitData()



    # # step3 create your model here , sending the parameters to create models

    # params ={"feature1":1,"feature2":1}
    # model = UserModel(params)


    # # step4 train an baseLine model
    # model.fit(1)
    # model.predict(train_data.data)
    # model.evaluate(eval_data.label,eval_data.data)



    # # step5 change tuned_params to choose a better model
    # # 在时间序列数据中，难以进行这种CV操作
    # # tuned_params = {"feature1":[1,2,3,4]}
    # # gs = GridSearchCV(UserModel(),tuned_params)
    # # gs.fit([1],[1])
    # # gs.best_params_










