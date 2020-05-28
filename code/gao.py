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

rcParams['figure.figsize'] = (13, 6)
# "STUDIO" "REDD
TEST_DATA = "STUDIO"
PRINT_STATS = False
PRINT_REPORT = False
IMPROVED = False
CROSS = False
TRAIN = False
INCLUDE_FAKE_BREAKPOINTS = False


# TODO: double check this i saw negative values
# TODO: further disaggregate appliances.
def concat_houses(data, houses_list=None, window_selection=None, include_fake_breakpoint=False, include_syntetic_data=False):
    _x1, _y1, _x2, _y2 = None, None, None, None
    if window_selection is None:
        window_selection = {1: (None, None)}
    if houses_list is None:
        houses_list = [1]

    for i in range(0, len(houses_list)):
        house = houses_list[i]
        print("loading house: " + str(house))
        selection = constants.selection_of_houses[house]
        window_start, window_end = window_selection[house]
        data.set_window(start=window_start, end=window_end)
        elec = data.buildings[house].elec

        train_appliances = dr.load_appliances_selection(elec, constants.order_appliances, selection,
                                                        constants.REDD_SAMPLE_PERIOD)
        train_total = dr.load_total_power_consumption(elec, selection, constants.REDD_SAMPLE_PERIOD)
        signals = Signals(constants.REDD_SAMPLE_PERIOD, constants.order_appliances, constants.breakpoint_classification,
                          IMPROVED, "temperature_redd.csv")
        signals.set_signals(train_appliances, train_total)
        x1_part = signals.get_input_bi()
        y1_part = signals.get_breakpoints()
        x2_part = signals.get_input_sl()
        y2_part = signals.get_labels()

        if include_fake_breakpoint:
            # 50 50 distribution
            fake_breakpoints = np.zeros(len(y1_part))
            br_count = np.count_nonzero(y1_part)
            fake_breakpoints[:br_count] = 1
            np.random.shuffle(fake_breakpoints)
            x2_fake = signals.get_input_sl_custom(fake_breakpoints)
            y2_fake = signals.get_labels_custom(fake_breakpoints)
            x2_part = np.concatenate((x2_part, x2_fake))
            y2_part = np.concatenate((y2_part, y2_fake))


        _x1 = x1_part if _x1 is None else np.concatenate((_x1, x1_part))
        _y1 = y1_part if _y1 is None else np.concatenate((_y1, y1_part))
        _x2 = x2_part if _x2 is None else np.concatenate((_x2, x2_part))
        _y2 = y2_part if _y2 is None else np.concatenate((_y2, y2_part))

    return _x1, _y1, _x2, _y2


def load_studio_data(_appliances, include_fake_breakpoint=False):
    ord_appliances = list(_appliances)
    signals = Signals(constants.STUDIO_SAMPLE_PERIOD, ord_appliances, constants.breakpoint_classification_my_data,
                      IMPROVED, "temperature_mydata.csv")
    signals.set_signals_mydata(_appliances)

    # signals.save_stats("studio_data_characteristics.csv")

    _x1 = signals.get_input_bi()
    _y1 = signals.get_breakpoints()
    _x2 = signals.get_input_sl()
    _y2 = signals.get_labels()

    if include_fake_breakpoint:
        # 50 50 distribution
        fake_breakpoints = np.zeros(len(_y1))
        br_count = np.count_nonzero(_y1)
        fake_breakpoints[:br_count] = 1
        np.random.shuffle(fake_breakpoints)
        x2_fake = signals.get_input_sl_custom(fake_breakpoints)
        y2_fake = signals.get_labels_custom(fake_breakpoints)
        _x2 = np.concatenate((_x2, x2_fake))
        _y2 = np.concatenate((_y2, y2_fake))

    return _x1, _y1, _x2, _y2


assert TEST_DATA == "REDD" or TEST_DATA == "STUDIO"
if TEST_DATA == "REDD":
    print("############################")
    print("####### loading REDD #######")
    print("############################")
    train = DataSet("../data/redd.5h")
    test = DataSet("../data/redd.5h")
    complete = DataSet("../data/redd.5h")
    houses = [1, 2, 3, 4]

    x1_train, y1_train, x2_train, y2_train = concat_houses(train, houses, constants.window_selection_of_houses,
                                                           INCLUDE_FAKE_BREAKPOINTS)
    x1_test, y1_test, x2_test, y2_test = concat_houses(test, [1], constants.window_selection_of_houses_test)
    x1, y1, x2, y2 = concat_houses(complete, houses, constants.window_selection_of_houses_complete)

else:
    print("############################")
    print("###### loading STUDIO ######")
    print("############################")
    appliances = dr.load_own_power_usage_data("studio_data.csv", constants.STUDIO_SAMPLE_PERIOD)
    split = int((len(appliances) * 3) / 4)
    x1_train, y1_train, x2_train, y2_train = load_studio_data(appliances.iloc[:split],
                                                              INCLUDE_FAKE_BREAKPOINTS)
    x1_test, y1_test, x2_test, y2_test = load_studio_data(appliances.iloc[split:])
    x1, y1, x2, y2 = load_studio_data(appliances)


#TEMP DELETE THIS
ord_appliances = list(appliances)
signals = Signals(constants.STUDIO_SAMPLE_PERIOD, ord_appliances, constants.breakpoint_classification_my_data,
                      IMPROVED, "temperature_mydata.csv")
test_arr = signals.set_signals_as_syntetic_mydata(appliances)

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
