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

from signals import Signals
from constants import breakpoint_classification, selection_of_houses, REDD_SAMPLE_PERIOD


def parse_input_mutli_appliances(x, y, number_of_appliances):
    multi_label = number_of_appliances + 1
    result = []
    other_states = []

    for _i in range(1, len(y) - 1):
        if y[_i] == multi_label:
            X = x[_i].copy()
            Z = []
            left = _i - 1
            right = _i + 1
            while y[left] == multi_label and left > 0:
                left -= 1
            while y[right] == multi_label and right < len(y):
                right += 1

            assert (not y[left] == multi_label) and (not y[right] == multi_label)

            # we subtract the average power of the base signal from the multiple appliance segment
            average_avg = 0
            average_min = 0
            if x[_i][0] > x[left][0]:
                average_avg = x[left][0]
                average_min = x[left][1]
                Z.append(y[left])

            if x[_i][0] > x[right][0]:
                average_avg += x[right][0]
                average_min += x[right][1]
                Z.append(y[right])

            # we subtract both neighbors from the segment
            if average_avg == x[right][0] + x[left][0] and y[left] == y[right]:
                average_avg = average_avg / 2
                average_min = average_min / 2

            X[0] -= average_avg
            X[1] -= average_min if average_min < x[_i][1] else x[_i][1]

            # TODO: change std as well

            result.append(X)
            other_states.append(np.unique(Z))

    return result, other_states


class Tester:
    def __init__(self, signal: Signals, order_appliances: [], sample_period: int):
        rcParams['figure.figsize'] = (13, 6)
        self.signal = signal
        self.sample_period = sample_period
        self.order_appliances = order_appliances

    def test_breakpoint_identifier(self, clf: MLPClassifier):
        appliances = self.signal.get_signals()
        states = np.array(self.signal.get_states())
        x1 = self.signal.get_input_bi()
        y1 = self.signal.get_breakpoints()

        pred_y1 = clf.predict(x1)

        print("breakpoint classifier")
        print(classification_report(y1, pred_y1))
        print(confusion_matrix(y1, pred_y1))

        accuracy_bi = accuracy_score(y1, pred_y1)
        print("accuracy bi: " + str(accuracy_bi))

        test_appliance_breakpoints = states.T
        nr_breakpoints = np.count_nonzero(y1)
        for br, ap in zip(test_appliance_breakpoints, appliances):
            ap["breakpoints"] = br * 400

        print("### breakpoints ###")
        print("Total Breakpoints:")
        print(nr_breakpoints)
        print("Plotting breakpoints per appliance")
        range_plot = [(0, int(25000 / self.sample_period)),
                      (int(24000 / self.sample_period), int(30000 / self.sample_period)),
                      (int(165000 / self.sample_period), int(172500 / self.sample_period)),
                      (int(335000 / self.sample_period), int(350000 / self.sample_period))]
        for ap, (x, y) in zip(appliances, range_plot):
            plt.plot(ap[x:y])
            plt.show()

    def test_segment_labeler(self, clf: MLPClassifier):
        y2 = self.signal.get_labels()
        x2 = self.signal.get_input_sl()
        pred_y2 = clf.predict(x2)

        print("segment labeler")
        print(classification_report(y2, pred_y2))
        print(confusion_matrix(y2, pred_y2))
        accuracy_sl = accuracy_score(y2, pred_y2)
        print("accuracy sl: " + str(accuracy_sl))

        (unique, counts) = np.unique(y2, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        print("### labels ###")
        print("total occurences of labels")
        print(frequencies)
        print("total predicted occurences of labels")
        print(np.asarray(np.unique(pred_y2, return_counts=True)).T)

        correct_labels = dp.breakpoint_states_to_labels(self.signal.get_states_on_breakpoints())
        actual_labels_without_multi = 0
        count_correct_labels_without_multi = 0
        amount_of_breakpoints = 0
        total_wrong = 0
        for t, c in zip(pred_y2, correct_labels):
            if not (t == len(self.order_appliances) + 1):
                amount_of_breakpoints += 1
                if len(c) == 0:
                    actual_labels_without_multi += 1
                    count_correct_labels_without_multi += 1 if t == 0 else 0
                    total_wrong += 0 if t == 0 else 1
                else:
                    actual_labels_without_multi += len(c)
                    count_correct_labels_without_multi += 1 if t in c else 0
                    total_wrong += 0 if t in c else 1
        print("from the total of " + str(amount_of_breakpoints))
        print("predicted " + str(count_correct_labels_without_multi) + " out of " + str(
            actual_labels_without_multi) + " appliances labels correctly")
        print("total wrong = " + str(total_wrong))

        print(" ")
        print("### segments ###")
        print("total segments")
        print(len(self.signal.get_segments()))

    def test_segment_labeler_custom(self, bi_clf: MLPClassifier, sl_clf: MLPClassifier):
        x1 = self.signal.get_input_bi()

        breakpoints = bi_clf.predict(x1)
        y2 = self.signal.get_labels_custom(breakpoints)
        x2 = self.signal.get_input_sl_custom(breakpoints)
        pred_y2 = sl_clf.predict(x2)

        print("\n\nsegment labeler from breakpoint identifier output")
        print(classification_report(y2, pred_y2))
        print(confusion_matrix(y2, pred_y2))
        accuracy_sl = accuracy_score(y2, pred_y2)
        print("accuracy sl: " + str(accuracy_sl))

        (unique, counts) = np.unique(y2, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        print("### labels ###")
        print("total occurences of labels")
        print(frequencies)
        print("total predicted occurences of labels")
        print(np.asarray(np.unique(pred_y2, return_counts=True)).T)

        states = self.signal.get_states()
        states_on_breakpoints = []
        for br, st in zip(breakpoints, states):
            if br == 1:
                states_on_breakpoints.append(st)
        correct_labels = dp.breakpoint_states_to_labels(states_on_breakpoints)
        actual_labels_without_multi = 0
        count_correct_labels_without_multi = 0
        amount_of_breakpoints = 0
        total_wrong = 0
        for t, c in zip(pred_y2, correct_labels):
            if not (t == len(self.order_appliances) + 1):
                amount_of_breakpoints += 1
                if len(c) == 0:
                    actual_labels_without_multi += 1
                    count_correct_labels_without_multi += 1 if t == 0 else 0
                    total_wrong += 0 if t == 0 else 1
                else:
                    actual_labels_without_multi += len(c)
                    count_correct_labels_without_multi += 1 if t in c else 0
                    total_wrong += 0 if t in c else 1
        print("from the total of " + str(amount_of_breakpoints))
        print("predicted " + str(count_correct_labels_without_multi) + " out of " + str(
            actual_labels_without_multi) + " appliances labels correctly")
        print("total wrong = " + str(total_wrong))

        print(" ")
        print("### segments ###")
        print("total segments")
        print(len(self.signal.get_segments()))

    def test_multi_appliance_dissagregator(self, clf: MLPClassifier):
        y2 = self.signal.get_labels()
        x2 = self.signal.get_input_sl()
        test_input3, other_labels = parse_input_mutli_appliances(x2, y2, len(self.order_appliances))
        new_labels = clf.predict(test_input3)
        correct_labels = dp.breakpoint_states_to_labels(self.signal.get_states_on_breakpoints())

        count_multi = 0
        count_labels = 0
        count_correct_labels_with_multi = 0
        total_wrong = 0

        for i, n in enumerate(np.where(np.array(y2) == len(self.order_appliances) + 1)[0]):
            count_multi += 1
            t = np.unique(np.append(other_labels[i], new_labels[i]))
            c = correct_labels[n]
            count_labels += len(c)
            for _t in t:
                count_correct_labels_with_multi += 1 if _t in c else 0
                total_wrong += 0 if _t in c else 1

        print("### Multi appliance ###")
        print("count multi: " + str(count_multi))
        print("total correct = " + str(count_labels) + " total predicted= " + str(
            count_correct_labels_with_multi))
        print("total wrong = " + str(total_wrong))

        count_multi = 0
        count_labels = 0
        count_correct_labels_with_multi = 0
        total_wrong = 0

        pred_y2 = clf.predict(x2)
        test_input3, other_labels = parse_input_mutli_appliances(x2, pred_y2, len(self.order_appliances))
        new_labels = clf.predict(test_input3)
        for i, n in enumerate(np.where(np.array(pred_y2) == len(self.order_appliances) + 1)[0]):
            count_multi += 1
            t = np.unique(np.append(other_labels[i], new_labels[i]))
            c = correct_labels[n]
            count_labels += len(c)
            for _t in t:
                count_correct_labels_with_multi += 1 if _t in c else 0
                total_wrong += 0 if _t in c else 1

        print(" ")
        print("combined with segment labeler")
        print("count multi: " + str(count_multi))
        print("total correct = " + str(count_labels) + " total predicted= " + str(
            count_correct_labels_with_multi))
        print("total wrong " + str(total_wrong))

    def test_cross_validation_bi(self, clf: MLPClassifier):
        x1 = self.signal.get_input_bi()
        y1 = self.signal.get_breakpoints()
        scores_breakpoints = cross_val_score(clf, x1, y1, cv=10)
        print("Accuracy breakpoints: %0.2f (+/- %0.2f)" % (scores_breakpoints.mean(), scores_breakpoints.std() * 2))

    def test_cross_validation_sl(self, clf: MLPClassifier):
        y2 = self.signal.get_labels()
        x2 = self.signal.get_input_sl()
        scores_labeler = cross_val_score(clf, x2, y2, cv=10)
        print("Accuracy labels: %0.2f (+/- %0.2f)" % (scores_labeler.mean(), scores_labeler.std() * 2))


def main():
    label_clf = load('../models/redd/segmentlabeler0.8646616541353384.ml')
    breakpoint_clf = load('../models/redd/breakpointidentifier0.9641493055555556.ml')
    test = DataSet("../data/redd.5h")
    start = datetime(2011, 5, 12, tzinfo=pytz.UTC)
    end = datetime(2011, 5, 20, tzinfo=pytz.UTC)
    test.set_window(start=start.strftime("%Y-%m-%d %H:%M:%S"), end=end.strftime("%Y-%m-%d %H:%M:%S"))
    test_elec = test.buildings[1].elec
    selection = selection_of_houses[1]
    order_appliances = ['fridge', 'microwave', 'washer dryer',
                        'dish washer']
    test_appliances = dr.load_appliances_selection(test_elec, order_appliances, selection, REDD_SAMPLE_PERIOD)
    test_total = dr.load_total_power_consumption(test_elec, selection, REDD_SAMPLE_PERIOD)
    test_signals = Signals(REDD_SAMPLE_PERIOD, order_appliances, breakpoint_classification)
    test_signals.set_signals(test_appliances, test_total)

    tester = Tester(test_signals, order_appliances, REDD_SAMPLE_PERIOD)
    tester.test_breakpoint_identifier(breakpoint_clf)
    tester.test_segment_labeler(label_clf)
    tester.test_multi_appliance_dissagregator(label_clf)
    tester.test_segment_labeler_custom(breakpoint_clf, label_clf)


if __name__ == "__main__":
    main()
