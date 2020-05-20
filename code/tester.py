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
from constants import breakpoint_classification, order_appliances, selection_of_houses, SAMPLE_PERIOD


class Tester:
    def __init__(self, signal: Signals, sample_period: int):
        rcParams['figure.figsize'] = (13, 6)
        self.signal = signal
        self.sample_period = sample_period

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
        print("distinct labels after multi appliance dissagregation:")
        # print(np.asarray(np.unique(addedLabels, return_counts=True)).T)

        print(" ")
        print("### segments ###")
        print("total segments")
        print(len(self.signal.get_segments()))

    def test_multi_appliance_dissagregator(self, clf: MLPClassifier):
        print("not implemented yet")
        # test_input3, other_labels = parse_input_mutli_appliances(x2_test, y2_test, len(order_appliances))
        # new_labels = label_clf.predict(test_input3)
        """correct_labels = dp.breakpoint_states_to_labels(test_signals.get_states_on_breakpoints())

        actual_labels = 0
        actual_labels_without_multi = 0
        actual_labels_with_multi = 0
        count_correct_labels_without_multi = 0
        count_correct_labels_with_multi = 0
        for t, c in zip(pred_y2, correct_labels):
            # print("got: "+ str(t)+ " wanted: "+str(c))
            actual_labels += len(c)
            if not (t == len(order_appliances) + 1):
                if len(c) == 0:
                    actual_labels_without_multi += 1
                    count_correct_labels_without_multi += 1 if t == 0 else 0
                else:
                    actual_labels_without_multi += len(c)
                    count_correct_labels_without_multi += 1 if t in c else 0

        for i, n in enumerate(np.where(np.array(y2_test) == 4)[0]):
            t = np.unique(np.append(other_labels[i], new_labels[i]))
            c = correct_labels[n]
            actual_labels_with_multi += len(c)
            for _t in t:
                count_correct_labels_with_multi += 1 if _t in c else 0

        print("actual labels = " + str(actual_labels))
        print("total correct = " + str(actual_labels_without_multi) + " total predicted= " + str(
            count_correct_labels_without_multi))
        print("total correct multi = " + str(actual_labels_with_multi) + " total predicted= " + str(
            count_correct_labels_with_multi))"""

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
    label_clf = load('../models/segmentlabeler0.8646616541353384.ml')
    breakpoint_clf = load('../models/breakpointidentifier0.9641493055555556.ml')
    test = DataSet("../data/redd.5h")
    start = datetime(2011, 5, 12, tzinfo=pytz.UTC)
    end = datetime(2011, 5, 20, tzinfo=pytz.UTC)
    test.set_window(start=start.strftime("%Y-%m-%d %H:%M:%S"), end=end.strftime("%Y-%m-%d %H:%M:%S"))
    test_elec = test.buildings[1].elec
    selection = selection_of_houses[1]
    test_appliances = dr.load_appliances_selection(test_elec, order_appliances, selection, SAMPLE_PERIOD)
    test_total = dr.load_total_power_consumption(test_elec, selection, SAMPLE_PERIOD)
    test_signals = Signals(SAMPLE_PERIOD, order_appliances, breakpoint_classification)
    test_signals.set_signals(test_appliances, test_total)

    tester = Tester(test_signals, SAMPLE_PERIOD)
    tester.test_breakpoint_identifier(breakpoint_clf)
    tester.test_segment_labeler(label_clf)


if __name__ == "__main__":
    main()
