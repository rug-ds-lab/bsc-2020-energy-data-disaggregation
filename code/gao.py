import sys
from datetime import datetime

import numpy as np
import pytz
from matplotlib import rcParams
from nilmtk import DataSet
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from datareader import Datareader as dr
from dataprepairer import Dataprepairer as dp
import pandas as pd
from joblib import dump, load

from imblearn.over_sampling import RandomOverSampler
from signals import Signals
from constants import SAMPLE_PERIOD, order_appliances, breakpoint_classification, selection_of_houses, window_selection_of_houses
from tester import Tester

rcParams['figure.figsize'] = (13, 6)
PRINT_STATS = False
PRINT_REPORT = False

#TODO: double check this i saw negative values
#TODO: further dissagregate appliances.

train = DataSet("../data/redd.5h")
test = DataSet("../data/redd.5h")

train_start = datetime(2011, 4, 18, 10, tzinfo=pytz.UTC)
train_end = datetime(2011, 5, 1, tzinfo=pytz.UTC)

test_start = datetime(2011, 5, 12, tzinfo=pytz.UTC)
test_end = datetime(2011, 5, 20, tzinfo=pytz.UTC)

train.set_window(start=train_start.strftime("%Y-%m-%d %H:%M:%S"), end=train_end.strftime("%Y-%m-%d %H:%M:%S"))
test.set_window(start=test_start.strftime("%Y-%m-%d %H:%M:%S"), end=test_end.strftime("%Y-%m-%d %H:%M:%S"))

houses = [1, 2, 3, 4]

# training
x1_train, y1_train, x2_train, y2_train = None, None, None, None

for i in range(0, len(houses)):
    house = houses[i]
    print("training house: " + str(house))
    selection = selection_of_houses[house]
    window_start, window_end = window_selection_of_houses[house]
    train.set_window(start=window_start, end=window_end)
    train_elec = train.buildings[house].elec

    train_appliances = dr.load_appliances_selection(train_elec, order_appliances, selection, SAMPLE_PERIOD)
    train_total = dr.load_total_power_consumption(train_elec, selection, SAMPLE_PERIOD)
    signals = Signals(SAMPLE_PERIOD, order_appliances, breakpoint_classification)
    signals.set_signals(train_appliances, train_total)
    x1_part = signals.get_input_bi()
    y1_part = signals.get_breakpoints()
    x2_part = signals.get_input_sl()
    y2_part = signals.get_labels()

    x1_train = x1_part if x1_train is None else np.concatenate((x1_train, x1_part))
    y1_train = y1_part if y1_train is None else np.concatenate((y1_train, y1_part))
    x2_train = x2_part if x2_train is None else np.concatenate((x2_train, x2_part))
    y2_train = y2_part if y2_train is None else np.concatenate((y2_train, y2_part))

house = 1
selection = selection_of_houses[house]
test_elec = test.buildings[house].elec
test_appliances = dr.load_appliances_selection(test_elec, order_appliances, selection, SAMPLE_PERIOD)
test_total = dr.load_total_power_consumption(test_elec, selection, SAMPLE_PERIOD)
test_signals = Signals(SAMPLE_PERIOD, order_appliances, breakpoint_classification)
test_signals.set_signals(test_appliances, test_total)

x1_test = test_signals.get_input_bi()
y1_test = test_signals.get_breakpoints()
x2_test = test_signals.get_input_sl()
y2_test = test_signals.get_labels()

best_bi_model, best_sl_model = None, None
best_accuracy_bi, best_accuracy_sl = 0, 0

breakpoint_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu", learning_rate="adaptive",
                                   max_iter=500)
label_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu", learning_rate="adaptive")

scores_breakpoints = cross_val_score(breakpoint_clf, x1_train, y1_train, cv=10)
print("Accuracy breakpoints: %0.2f (+/- %0.2f)" % (scores_breakpoints.mean(), scores_breakpoints.std() * 2))
scores_label = cross_val_score(label_clf, x2_train, y2_train, cv=10)
print("Accuracy label: %0.2f (+/- %0.2f)" % (scores_label.mean(), scores_label.std() * 2))

for i in range(0, 20):
    breakpoint_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu", learning_rate="adaptive",
                                   max_iter=500)
    label_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu", learning_rate="adaptive")

    multi_appliance_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu", learning_rate="adaptive")

    ros = RandomOverSampler(random_state=0)
    x1_resampled, y1_resampled = ros.fit_resample(x1_train, y1_train)
    print("training breakpoint identifier on iteration: " + str(i))
    breakpoint_clf.fit(x1_resampled, y1_resampled)
    x2_resampled, y2_resampled = ros.fit_resample(x2_train, y2_train)
    print("training segment labeler on iteration: " + str(i))
    label_clf.fit(x2_resampled, y2_resampled)

    pred_y1 = breakpoint_clf.predict(x1_test)
    pred_y2 = label_clf.predict(x2_test)

    if PRINT_REPORT:
        print("breakpoint classifier")
        print(classification_report(y1_test, pred_y1))
        print(confusion_matrix(y1_test, pred_y1))
    accuracy_bi = accuracy_score(y1_test, pred_y1)
    print("accuracy bi: " + str(accuracy_bi))

    if PRINT_REPORT:
        print("segment labeler")
        print(classification_report(y2_test, pred_y2))
        print(confusion_matrix(y2_test, pred_y2))
    accuracy_sl = accuracy_score(y2_test, pred_y2)
    print("accuracy sl: "+str(accuracy_sl))

    if accuracy_bi > best_accuracy_bi:
        best_accuracy_bi = accuracy_bi
        best_bi_model = breakpoint_clf

    if accuracy_sl > best_accuracy_sl:
        best_accuracy_sl = accuracy_sl
        best_sl_model = label_clf

dump(best_bi_model, '../models/breakpointidentifier'+str(best_accuracy_bi)+".ml")
dump(best_sl_model, '../models/segmentlabeler'+str(best_accuracy_sl)+".ml")

def main(argv):
    print("empty")


if __name__ == "__main__":
    main(sys.argv[1:])

