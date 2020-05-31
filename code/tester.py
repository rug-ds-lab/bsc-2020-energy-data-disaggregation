from datetime import datetime

import numpy as np
import pytz
from matplotlib import rcParams
from nilmtk import DataSet
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
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


class Tester:
    def __init__(self, signal: Signals, order_appliances: [], sample_period: int):
        rcParams['figure.figsize'] = (13, 6)
        self.signal = signal
        self.sample_period = sample_period
        self.order_appliances = order_appliances

    def test_breakpoint_identifier(self, clf: MLPClassifier, plot=False):
        appliances = self.signal.get_signals()
        states = np.array(self.signal.get_states())
        x1 = self.signal.get_input_bi()
        y1 = self.signal.get_breakpoints()

        pred_y1 = clf.predict(x1)

        print("breakpoint classifier")
        class_report = classification_report(y1, pred_y1, output_dict=True)
        print(class_to_latex(class_report, ["non-breakpoint", "breakpoint"]))
        print(" ")
        # print(classification_report(y1, pred_y1))

        print(conf_to_latex(confusion_matrix(y1, pred_y1), ["non-breakpoint", "breakpoint"]))
        # print(confusion_matrix(y1, pred_y1))

        accuracy_bi = accuracy_score(y1, pred_y1)
        print("accuracy bi: " + str(accuracy_bi))

        test_appliance_breakpoints = states.T
        nr_breakpoints = np.count_nonzero(y1)

        print("### breakpoints ###")
        print("Total Breakpoints:")
        print(nr_breakpoints)
        if plot:
            for br, ap in zip(test_appliance_breakpoints, appliances):
                ap["breakpoints"] = br * 400
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
        class_report = classification_report(y2, pred_y2, output_dict=True)
        print(class_to_latex(class_report, ["empty"] + self.order_appliances + ["multi"]))
        print(" ")
        # print(classification_report(y2, pred_y2))

        seen_labels = np.array(["empty"] + self.order_appliances + ["multi"])[
            np.array(list(class_report.keys())[:-3]).astype(int)]
        print(conf_to_latex(confusion_matrix(y2, pred_y2), seen_labels))
        # print(confusion_matrix(y2, pred_y2))
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
        class_report = classification_report(y2, pred_y2, output_dict=True)
        print(class_to_latex(class_report, ["empty"] + self.order_appliances + ["multi"]))
        # print(classification_report(y2, pred_y2))

        seen_labels = np.array(["empty"] + self.order_appliances + ["multi"])[
            np.array(list(class_report.keys())[:-3]).astype(int)]
        print(conf_to_latex(confusion_matrix(y2, pred_y2), seen_labels))
        # print(confusion_matrix(y2, pred_y2))

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
        print("\n")

    def test_multi_appliance_dissagregator_custom(self, bi_clf: MLPClassifier, sl_clf: MLPClassifier, improved: bool):
        print("######      custom       ######")
        x1 = self.signal.get_input_bi()
        breakpoints = bi_clf.predict(x1)
        x2 = self.signal.get_input_sl_custom(breakpoints)
        labels = sl_clf.predict(x2)
        self.test_multi_appliance_dissagregator(sl_clf, improved, breakpoints, labels)

    def test_multi_appliance_dissagregator(self, clf: MLPClassifier, improved: bool, breakpoints=None, labels=None):
        print("### multi appliance labeler ###")
        multi_dissagregator = Multi_dissagregator(self.signal, clf, self.order_appliances, self.sample_period, improved,
                                                  breakpoints, labels)
        multi_count = multi_dissagregator.count_multi_consumption_per_appliance()
        prev_count = 0
        print("start count mult: " + str(multi_count[0]))

        while not multi_count == prev_count:
            multi_dissagregator.diss_signal()
            prev_count = multi_count
            multi_count = multi_dissagregator.count_multi_consumption_per_appliance()
            print("after iteration: " + str(multi_count[0]))

        correct, total, correct_multi, total_multi, multi_count_total = multi_dissagregator.get_statts()
        print("multi count: " + str(multi_count_total))
        print("correct: " + str(correct))
        print("total: " + str(total))
        print("correct multi: " + str(correct_multi))
        print("total multi: " + str(total_multi))
        print("\n")

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
    """ you may set IMPROVED to either True or False, depending on if you want to test the improved version or not
        you may set TEST_DATA to either 'REDD' or 'STUDIO' depening on if you want to test
        on the REDD dataset or the STUDIO dataset"""
    IMPROVED = True
    # "STUDIO" "REDD"
    TEST_DATA = "GEN"

    subfolder = "improved" if IMPROVED else "original"
    label_clf = load('../models/' + TEST_DATA + '/' + subfolder + '/segmentlabeler.ml')
    breakpoint_clf = load('../models/' + TEST_DATA + '/' + subfolder + '/breakpointidentifier.ml')

    assert TEST_DATA == "REDD" or TEST_DATA == "STUDIO" or TEST_DATA == "GEN"
    sample_period = constants.SAMPLE_PERIOD[TEST_DATA]
    if TEST_DATA == "REDD":
        order_appliances = constants.order_appliances

        signals = REDDloader(constants.window_selection_of_houses_test, constants.selection_of_appliances,
                             order_appliances, sample_period, IMPROVED).load_house(1)
    else:
        appliances = dr.load_own_power_usage_data("../data/studio_data.csv", sample_period)
        order_appliances = list(appliances) if TEST_DATA == "STUDIO" else constants.order_appliances_gen_STUDIO
        split = int((len(appliances) * 3) / 4) if TEST_DATA == "STUDIO" else None
        signals = STUDIOloader(sample_period, IMPROVED, appliances=appliances, split=(split, None),
                               order_appliances=order_appliances).get_signals()


    tester = Tester(signals, order_appliances, sample_period)
    tester.test_breakpoint_identifier(breakpoint_clf)
    tester.test_segment_labeler(label_clf)
    tester.test_segment_labeler_custom(breakpoint_clf, label_clf)
    tester.test_multi_appliance_dissagregator(label_clf, IMPROVED)
    tester.test_multi_appliance_dissagregator_custom(breakpoint_clf, label_clf, IMPROVED)


if __name__ == "__main__":
    main()
