from datetime import datetime

import numpy as np
import pytz
from matplotlib import rcParams
from nilmtk import DataSet
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from dataprepairer import Dataprepairer as dp
import pandas as pd
from joblib import load

from signals import Signals
from multiApplianceDisaggregator import Multi_dissagregator
import constants
from REDDloader import REDDloader
from STUDIOloader import STUDIOloader
from datareader import Datareader as dr
from LatexConverter import class_to_latex, conf_to_latex

is_improved = True
test_data = "REDD"

subfolder = "improved" if is_improved else "original"
label_clf = load('../models/' + test_data + '/' + subfolder + '/segmentlabeler.ml')
print(label_clf.coefs_[0].mean(axis=1))
