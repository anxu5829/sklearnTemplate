import numpy as np

def userDefinedLoss(actual,pred):
    return np.sum(actual - pred)



def focalLoss(preds,train_data):
    labels = train_data.get_label()
    pass