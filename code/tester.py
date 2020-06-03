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


class Tester:
    def __init__(self, test_data: str, is_improved: bool):
        rcParams['figure.figsize'] = (13, 6)
        assert test_data == "REDD" or test_data == "STUDIO" or test_data == "GEN"

        self.sample_period = constants.SAMPLE_PERIOD[test_data]
        self.test_data = test_data
        self.is_improved = is_improved
        self.signal, self.order_appliances = self.load_data()
        subfolder = "improved" if self.is_improved else "original"
        self.label_clf = load('../models/' + self.test_data + '/' + subfolder + '/segmentlabeler.ml')
        self.breakpoint_clf = load('../models/' + self.test_data + '/' + subfolder + '/breakpointidentifier.ml')

        file_path = "../tests/" + self.test_data + "/" + subfolder + "/test" + datetime.now().strftime(
            "%d_%m_%Y_%H_%M") + ".txt"
        self.file = open(file_path, "w")

    def __del__(self):
        self.file.close()

    def load_data(self):
        if self.test_data == "REDD":
            print("LOADING REDD")
            order_appliances = constants.order_appliances
            signals = REDDloader(constants.window_selection_of_houses_test, constants.selection_of_appliances,
                                 order_appliances, self.sample_period, self.is_improved).load_house(1)
        elif self.test_data == "STUDIO":
            print("LOADING STUDIO")
            appliances = dr.load_own_power_usage_data("../data/studio_data.csv", self.sample_period)
            order_appliances = list(appliances)
            split = int((len(appliances) * 3) / 4)
            signals = STUDIOloader(self.sample_period, self.is_improved, appliances=appliances, split=(split, None),
                                   order_appliances=order_appliances).get_signals()
        else:
            print("LOADING GEN")
            appliances = dr.load_own_power_usage_data("../data/studio_data.csv", self.sample_period)
            order_appliances = constants.order_appliances_gen_STUDIO
            signals = STUDIOloader(self.sample_period, self.is_improved, appliances=appliances, split=(None, None),
                                   order_appliances=order_appliances).get_signals()

        return signals, order_appliances

    def test_breakpoint_identifier(self, plot=False):
        self.file.write("\n\nBREAKPOINT IDENTIFIER TEST\n")
        print("BREAKPOINT IDENTIFIER TEST")
        appliances = self.signal.get_signals()
        states = np.array(self.signal.get_states())
        x1 = self.signal.get_input_bi()
        y1 = self.signal.get_breakpoints()

        pred_y1 = self.breakpoint_clf.predict(x1)

        self.file.write("breakpoint classifier\n")
        class_report = classification_report(y1, pred_y1, output_dict=True)
        self.file.write(class_to_latex(class_report, ["non-breakpoint", "breakpoint"]))
        self.file.write("\n\n")

        self.file.write(conf_to_latex(confusion_matrix(y1, pred_y1), ["non-breakpoint", "breakpoint"]))
        self.file.write("\n\n")

        accuracy_bi = accuracy_score(y1, pred_y1)
        self.file.write("accuracy bi: " + str(accuracy_bi)+"\n")

        test_appliance_breakpoints = states.T
        nr_breakpoints = np.count_nonzero(y1)

        self.file.write("### breakpoints ###\n")
        self.file.write("Total Breakpoints:\n")
        self.file.write(str(nr_breakpoints)+"\n")
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

    def test_segment_labeler(self):
        self.file.write("\n\ntest segment labeler\n")
        print("SEGMENT LABELER TEST")
        y2 = self.signal.get_labels()
        x2 = self.signal.get_input_sl()
        pred_y2 = self.label_clf.predict(x2)

        self.file.write("segment labeler\n")
        class_report = classification_report(y2, pred_y2, output_dict=True)
        self.file.write(class_to_latex(class_report, ["empty"] + self.order_appliances + ["multi"]))
        self.file.write("\n\n")

        seen_labels = np.array(["empty"] + self.order_appliances + ["multi"])[
            np.array(list(class_report.keys())[:-3]).astype(int)]
        self.file.write(conf_to_latex(confusion_matrix(y2, pred_y2), seen_labels))
        self.file.write("\n\n")

        accuracy_sl = accuracy_score(y2, pred_y2)
        self.file.write("accuracy sl: " + str(accuracy_sl)+"\n")

        (unique, counts) = np.unique(y2, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        self.file.write("### labels ###\n")
        self.file.write("total occurences of labels\n")
        self.file.write(str(frequencies)+"\n")
        self.file.write("total predicted occurences of labels\n")
        self.file.write(str(np.asarray(np.unique(pred_y2, return_counts=True)).T)+"\n")

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
        self.file.write("from the total of " + str(amount_of_breakpoints)+"\n")
        self.file.write("predicted " + str(count_correct_labels_without_multi) + " out of " + str(
            actual_labels_without_multi) + " appliances labels correctly\n")
        self.file.write("total wrong = " + str(total_wrong)+"\n")

        self.file.write("\n")
        self.file.write("### segments ###\n")
        self.file.write("total segments\n")
        self.file.write(str(len(self.signal.get_segments()))+"\n")

    def test_segment_labeler_custom(self):
        self.file.write("\n\n##segment labeler from breakpoint identifier output##\n")
        print("SEGMENT LABELER TEST CUSTOM")
        x1 = self.signal.get_input_bi()

        breakpoints = self.breakpoint_clf.predict(x1)
        y2 = self.signal.get_labels_custom(breakpoints)
        x2 = self.signal.get_input_sl_custom(breakpoints)
        pred_y2 = self.label_clf.predict(x2)

        self.file.write("segment labeler from breakpoint identifier output\n")
        class_report = classification_report(y2, pred_y2, output_dict=True)
        self.file.write(class_to_latex(class_report, ["empty"] + self.order_appliances + ["multi"]))
        self.file.write("\n\n")

        seen_labels = np.array(["empty"] + self.order_appliances + ["multi"])[
            np.array(list(class_report.keys())[:-3]).astype(int)]
        self.file.write(conf_to_latex(confusion_matrix(y2, pred_y2), seen_labels))
        self.file.write("\n\n")

        accuracy_sl = accuracy_score(y2, pred_y2)
        self.file.write("accuracy sl: " + str(accuracy_sl)+"\n")

        (unique, counts) = np.unique(y2, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        self.file.write("### labels ###\n")
        self.file.write("total occurences of labels\n")
        self.file.write(str(frequencies)+"\n")
        self.file.write("total predicted occurences of labels\n")
        self.file.write(str(np.asarray(np.unique(pred_y2, return_counts=True)).T)+"\n")

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
        self.file.write("from the total of " + str(amount_of_breakpoints)+"\n")
        self.file.write("predicted " + str(count_correct_labels_without_multi) + " out of " + str(
            actual_labels_without_multi) + " appliances labels correctly\n")
        self.file.write("total wrong = " + str(total_wrong)+"\n")

        self.file.write("\n\n")
        self.file.write("### segments ###\n")
        self.file.write("total segments\n")
        self.file.write(str(len(self.signal.get_segments()))+"\n")
        self.file.write("\n\n")

    def test_multi_appliance_dissagregator_custom(self):
        self.file.write("\n\n######      custom       ######\n")
        print("MULTI TEST CUSTOM")
        x1 = self.signal.get_input_bi()
        breakpoints = self.breakpoint_clf.predict(x1)
        x2 = self.signal.get_input_sl_custom(breakpoints)
        labels = self.label_clf.predict(x2)
        self.test_multi_appliance_dissagregator(breakpoints, labels)

    def test_multi_appliance_dissagregator(self, breakpoints=None, labels=None):
        print("MULTI TEST")
        self.file.write("\n\n### multi appliance labeler ###\n")
        multi_dissagregator = Multi_dissagregator(self.signal, self.label_clf, self.order_appliances,
                                                  self.sample_period, self.is_improved,
                                                  breakpoints, labels)
        multi_count = multi_dissagregator.count_multi_consumption_per_appliance()
        prev_count = 0
        self.file.write("start count mult: " + str(multi_count[0])+"\n")

        while not multi_count == prev_count:
            multi_dissagregator.diss_signal()
            prev_count = multi_count
            multi_count = multi_dissagregator.count_multi_consumption_per_appliance()
            self.file.write("after iteration: " + str(multi_count[0])+"\n")

        correct, total, correct_multi, total_multi, multi_count_total = multi_dissagregator.get_statts()
        self.file.write("multi count: " + str(multi_count_total)+"\n")
        self.file.write("correct: " + str(correct)+"\n")
        self.file.write("total: " + str(total)+"\n")
        self.file.write("correct multi: " + str(correct_multi)+"\n")
        self.file.write("total multi: " + str(total_multi)+"\n")
        self.file.write("\n")

    def principal_component_analysis(self):
        self.file.write("\n\n PCA\n")
        print("PCA")
        pca = PCA()
        x2 = self.signal.get_input_sl()
        pca.fit(x2)
        self.file.write(str(pca.explained_variance_ratio_)+"\n")
        self.file.write(str(pca.singular_values_)+"\n")


def main():
    """ you may set IMPROVED to either True or False, depending on if you want to test the improved version or not
        you may set TEST_DATA to either 'REDD' or 'STUDIO' depening on if you want to test
        on the REDD dataset or the STUDIO dataset"""

    schedule = [{"test_data": "REDD", "is_improved": False},
                {"test_data": "STUDIO", "is_improved": False},
                {"test_data": "REDD", "is_improved": True},
                {"test_data": "STUDIO", "is_improved": True},
                {"test_data": "GEN", "is_improved": True}
    ]

    for batch in schedule:
        tester = Tester(batch["test_data"], batch["is_improved"])
        tester.test_breakpoint_identifier()
        tester.test_segment_labeler()
        tester.test_segment_labeler_custom()
        tester.test_multi_appliance_dissagregator()
        tester.test_multi_appliance_dissagregator_custom()
        tester.principal_component_analysis()


if __name__ == "__main__":
    main()
