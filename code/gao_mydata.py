import csv
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
from tester import Tester

SAMPLE_PERIOD = 10
rcParams['figure.figsize'] = (13, 6)
PRINT_REPORT = True
CROSS = False
TRAIN = False
TEST = True
IMPROVED = False

breakpoint_classification = [
    # tv
    {"max_power": 65, "on_power_threshold": 15, "min_on": int(300 / SAMPLE_PERIOD),
     "min_off": int(300 / SAMPLE_PERIOD)},
    # phone_charger
    {"max_power": 26, "on_power_threshold": 5, "min_on": int(60 / SAMPLE_PERIOD),
     "min_off": int(60 / SAMPLE_PERIOD)},
    # desk_lamp
    {"max_power": 37, "on_power_threshold": 10, "min_on": int(10 / SAMPLE_PERIOD),
     "min_off": int(10 / SAMPLE_PERIOD)},
    # couch_lamp
    {"max_power": 54, "on_power_threshold": 10, "min_on": int(10 / SAMPLE_PERIOD),
     "min_off": int(10 / SAMPLE_PERIOD)},
    # washing_machine
    {"max_power": 2067, "on_power_threshold": 100, "min_on": int(3600 / SAMPLE_PERIOD),
     "min_off": int(1200 / SAMPLE_PERIOD)},
    # fridge
    {"max_power": 704, "on_power_threshold": 25, "min_on": int(30 / SAMPLE_PERIOD),
     "min_off": int(12 / SAMPLE_PERIOD)},
    # water_heater
    {"max_power": 1989, "on_power_threshold": 1000, "min_on": int(1 / SAMPLE_PERIOD),
     "min_off": int(1 / SAMPLE_PERIOD)},
    # alienware_laptop
    {"max_power": 144, "on_power_threshold": 20, "min_on": int(60 / SAMPLE_PERIOD),
     "min_off": int(60 / SAMPLE_PERIOD)},
    # ps4
    {"max_power": 152, "on_power_threshold": 60, "min_on": int(60 / SAMPLE_PERIOD),
     "min_off": int(60 / SAMPLE_PERIOD)},
    # microwave
    {"max_power": 2510, "on_power_threshold": 1000, "min_on": int(30 / SAMPLE_PERIOD),
     "min_off": int(30 / SAMPLE_PERIOD)}
]

appliances = dr.load_own_power_usage_data("studio_data.csv", SAMPLE_PERIOD)
order_appliances = list(appliances)
signals = Signals(SAMPLE_PERIOD, order_appliances, breakpoint_classification, IMPROVED, "temperature_mydata.csv")
signals.set_signals_mydata(appliances)

# signals.save_stats("studio_data_characteristics.csv")
# signals.save_breakpoint_classification("studio_data_classification.csv")

train_signals = Signals(SAMPLE_PERIOD, order_appliances, breakpoint_classification, IMPROVED, "temperature_mydata.csv")
test_signals = Signals(SAMPLE_PERIOD, order_appliances, breakpoint_classification, IMPROVED, "temperature_mydata.csv")
split = int((len(appliances) * 3) / 4)
train_signals.set_signals_mydata(appliances.iloc[:split])
test_signals.set_signals_mydata(appliances.iloc[split:])

x1 = signals.get_input_bi()
y1 = signals.get_breakpoints()
x2 = signals.get_input_sl()
y2 = signals.get_labels()

x1_train = train_signals.get_input_bi()
y1_train = train_signals.get_breakpoints()
x2_train = train_signals.get_input_sl()
y2_train = train_signals.get_labels()

x1_test = test_signals.get_input_bi()
y1_test = test_signals.get_breakpoints()
x2_test = test_signals.get_input_sl()
y2_test = test_signals.get_labels()

best_bi_model, best_sl_model = None, None
best_accuracy_bi, best_accuracy_sl = 0, 0

if CROSS:
    breakpoint_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu", learning_rate="adaptive")
    label_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu", learning_rate="adaptive")
    print("calculating 10-fold cross validation score")
    scores_breakpoints = cross_val_score(breakpoint_clf, x1, y1, cv=10)
    print("Accuracy breakpoints: %0.2f (+/- %0.2f)" % (scores_breakpoints.mean(), scores_breakpoints.std() * 2))
    scores_label = cross_val_score(label_clf, x2, y2, cv=10)
    print("Accuracy label: %0.2f (+/- %0.2f)" % (scores_label.mean(), scores_label.std() * 2))

if TRAIN:
    for i in range(0, 30):
        breakpoint_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu", learning_rate="adaptive")
        label_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu", learning_rate="adaptive")

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
        print("accuracy sl: " + str(accuracy_sl))

        if accuracy_bi > best_accuracy_bi:
            best_accuracy_bi = accuracy_bi
            best_bi_model = breakpoint_clf

        if accuracy_sl > best_accuracy_sl:
            best_accuracy_sl = accuracy_sl
            best_sl_model = label_clf

    dump(best_bi_model, '../models/mydata/breakpointidentifier' + str(best_accuracy_bi) + ".ml")
    dump(best_sl_model, '../models/mydata/segmentlabeler' + str(best_accuracy_sl) + ".ml")

if TEST:
    label_clf = load('../models/mydata/segmentlabeler0.8989547038327527.ml')
    breakpoint_clf = load('../models/mydata/breakpointidentifier0.9555114638447971.ml')

    tester = Tester(test_signals, order_appliances, SAMPLE_PERIOD)
    tester.test_breakpoint_identifier(breakpoint_clf)
    tester.test_segment_labeler(label_clf)
    tester.test_multi_appliance_dissagregator(label_clf)
    tester.test_segment_labeler_custom(breakpoint_clf, label_clf)
