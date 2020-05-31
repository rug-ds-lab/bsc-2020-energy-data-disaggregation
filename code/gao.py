import sys

import numpy as np
from matplotlib import rcParams
from nilmtk import DataSet
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from datareader import Datareader as dr
import pandas as pd
from joblib import dump

from imblearn.over_sampling import RandomOverSampler
from signals import Signals
import constants
from REDDloader import REDDloader
from STUDIOloader import STUDIOloader

rcParams['figure.figsize'] = (13, 6)
# "STUDIO" "REDD" "GEN"
TEST_DATA = "GEN"
PRINT_STATS = False
PRINT_REPORT = True
IMPROVED = True
CROSS = False
TRAIN = True
INCLUDE_FAKE_BREAKPOINTS = True

assert TEST_DATA == "REDD" or TEST_DATA == "STUDIO" or TEST_DATA == "GEN"
sample_period = constants.SAMPLE_PERIOD[TEST_DATA]
if TEST_DATA == "REDD":
    print("############################")
    print("####### loading REDD #######")
    print("############################")
    houses = [1, 2, 3, 4]
    selection_of_appliances = constants.selection_of_appliances
    order_appliances = constants.order_appliances

    x1_train, y1_train, x2_train, y2_train = REDDloader(constants.window_selection_of_houses,
                                                        selection_of_appliances, order_appliances, sample_period,
                                                        IMPROVED).concat_houses(houses, INCLUDE_FAKE_BREAKPOINTS)
    x1_test, y1_test, x2_test, y2_test = REDDloader(constants.window_selection_of_houses_test,
                                                    selection_of_appliances, order_appliances, sample_period,
                                                    IMPROVED).concat_houses([1])
    x1, y1, x2, y2 = REDDloader(constants.window_selection_of_houses_complete,
                                selection_of_appliances, order_appliances, sample_period,
                                IMPROVED).concat_houses(houses)

elif TEST_DATA == "STUDIO":
    print("############################")
    print("###### loading STUDIO ######")
    print("############################")
    appliances = dr.load_own_power_usage_data("../data/studio_data.csv", sample_period)
    split = int((len(appliances) * 3) / 4)
    x1_train, y1_train, x2_train, y2_train = STUDIOloader(sample_period, IMPROVED,
                                                          appliances=appliances, split=(None, split)).load(
        INCLUDE_FAKE_BREAKPOINTS)
    x1_test, y1_test, x2_test, y2_test = STUDIOloader(sample_period, IMPROVED, appliances=appliances,
                                                      split=(split, None)).load()
    x1, y1, x2, y2 = STUDIOloader(sample_period, IMPROVED, appliances=appliances).load()

else:
    print("############################")
    print("# loading REDD and STUDIO  #")
    print("############################")
    train = DataSet("../data/redd.5h")
    houses = [1, 2, 3, 4]
    x1_train, y1_train, x2_train, y2_train = REDDloader(constants.window_selection_of_houses_complete,
                                                        constants.selection_of_generalizable_appliances,
                                                        constants.order_appliances_gen_REDD, sample_period,
                                                        IMPROVED).concat_houses(houses, INCLUDE_FAKE_BREAKPOINTS)
    x1_test, y1_test, x2_test, y2_test = STUDIOloader(sample_period, IMPROVED,
                                                      order_appliances=constants.order_appliances_gen_STUDIO).load()
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
    for i in range(0, 20):
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

    if IMPROVED:
        dump(best_bi_model, '../models/' + TEST_DATA + '/improved/breakpointidentifier2.ml')
        dump(best_sl_model, '../models/' + TEST_DATA + '/improved/segmentlabeler2.ml')
    else:
        dump(best_bi_model, '../models/' + TEST_DATA + '/original/breakpointidentifier2.ml')
        dump(best_sl_model, '../models/' + TEST_DATA + '/original/segmentlabeler2.ml')



    print("best accuracy bi: %0.2f" % best_accuracy_bi)
    print("best accuracy sl: %0.2f" % best_accuracy_sl)
