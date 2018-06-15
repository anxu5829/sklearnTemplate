import config
import os
import easydict as edict
from dataLoader import DataLoader,Data
from userDefinedModel import UserLightgbmModel
import logging
import argparse

train = Data()
train = train.load_data(os.getcwd())