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

from dataprepairer import Dataprepairer as Dp
import pandas as pd
from joblib import load

from signals import Signals
from multiApplianceDisaggregator import Multi_dissagregator
import constants
from REDDloader import REDDloader
from STUDIOloader import STUDIOloader
from datareader import Datareader as Dr
from LatexConverter import class_to_latex, conf_to_latex, multi_to_latex


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
        print("saving output in " + file_path)
        self.file = open(file_path, "w")
        if test_data == "REDD":
            self.dataname = "REDD"
        elif test_data == "STUDIO":
            self.dataname = "the studio data"
        else:
            self.dataname = "a combination of the 2 datasets"

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
            appliances = Dr.load_own_power_usage_data("../data/studio_data.csv", self.sample_period)
            order_appliances = list(appliances)
            split = int((len(appliances) * 3) / 4)
            signals = STUDIOloader(self.sample_period, self.is_improved, appliances=appliances, split=(split, None),
                                   order_appliances=order_appliances).get_signals()
        else:
            print("LOADING GEN")
            appliances = Dr.load_own_power_usage_data("../data/studio_data.csv", self.sample_period)
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
        accuracy_bi = accuracy_score(y1, pred_y1)
        caption = "of the" + (
            " improved " if self.is_improved else " ") + "breakpoint identifier using " + self.dataname
        label = "bi" + ("_improved_" if self.is_improved else "_") + self.test_data
        class_report = classification_report(y1, pred_y1, output_dict=True)
        self.file.write(class_to_latex(class_report, ["non-breakpoint", "breakpoint"], caption, label, accuracy_bi))
        self.file.write("\n\n")

        self.file.write(conf_to_latex(confusion_matrix(y1, pred_y1), ["non-breakpoint", "breakpoint"], caption, label))
        self.file.write("\n\n")
        self.file.write("accuracy bi: " + str(accuracy_bi) + "\n")

        test_appliance_breakpoints = states.T
        nr_breakpoints = np.count_nonzero(y1)

        self.file.write("### breakpoints ###\n")
        self.file.write("Total Breakpoints:\n")
        self.file.write(str(nr_breakpoints) + "\n")
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

    def test_segment_labeler(self, breakpoints=None, caption_add="", label_add=""):
        self.file.write("\n\ntest segment labeler\n")
        print("SEGMENT LABELER TEST")
        if breakpoints is None:
            y2 = self.signal.get_labels()
            x2 = self.signal.get_input_sl()
            segments = self.signal.get_segments()
            states_on_breakpoints = self.signal.get_states_on_breakpoints()
        else:
            y2 = self.signal.get_labels_custom(breakpoints)
            x2 = self.signal.get_input_sl_custom(breakpoints)
            segments = self.signal.get_segments_custom(breakpoints)
            states_on_breakpoints = self.signal.get_states_on_breakpoints_custom(breakpoints)
        pred_y2 = self.label_clf.predict(x2)

        self.file.write("segment labeler\n")
        accuracy_sl = accuracy_score(y2, pred_y2)
        caption = "of the" + (
            " improved " if self.is_improved else " ") + "segment labeler " + caption_add + "using " + self.dataname
        label = "sl" + ("_improved_" if self.is_improved else "_") + self.test_data + label_add
        class_report = classification_report(y2, pred_y2, output_dict=True)
        self.file.write(
            class_to_latex(class_report, ["empty"] + self.order_appliances + ["multi"], caption, label, accuracy_sl))
        self.file.write("\n\n")

        seen_labels = np.array(["empty"] + self.order_appliances + ["multi"])[
            np.array(list(class_report.keys())[:-3]).astype(int)]
        self.file.write(conf_to_latex(confusion_matrix(y2, pred_y2), seen_labels, caption, label))
        self.file.write("\n\n")
        self.file.write("accuracy sl: " + str(accuracy_sl) + "\n")

        (unique, counts) = np.unique(y2, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        self.file.write("### labels ###\n")
        self.file.write("total occurences of labels\n")
        self.file.write(str(frequencies) + "\n")
        self.file.write("total predicted occurences of labels\n")
        self.file.write(str(np.asarray(np.unique(pred_y2, return_counts=True)).T) + "\n")

        correct_labels = Dp.breakpoint_states_to_labels(states_on_breakpoints)
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
        self.file.write("from the total of " + str(amount_of_breakpoints) + "\n")
        self.file.write("predicted " + str(count_correct_labels_without_multi) + " out of " + str(
            actual_labels_without_multi) + " appliances labels correctly\n")
        self.file.write("total wrong = " + str(total_wrong) + "\n")

        self.file.write("\n")
        self.file.write("### segments ###\n")
        self.file.write("total segments\n")
        self.file.write(str(len(segments)) + "\n")

    def test_segment_labeler_custom(self):
        self.file.write("\n\n##segment labeler from breakpoint identifier output##\n")
        print("SEGMENT LABELER TEST CUSTOM")
        x1 = self.signal.get_input_bi()
        breakpoints = self.breakpoint_clf.predict(x1)
        caption_add = "with breakpoints given by the breakpoint identifier "
        label_add = "_custom"
        self.test_segment_labeler(breakpoints, caption_add, label_add)

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
        self.file.write("start count mult: " + str(multi_count[0]) + "\n")

        while not multi_count == prev_count:
            multi_dissagregator.diss_signal()
            prev_count = multi_count
            multi_count = multi_dissagregator.count_multi_consumption_per_appliance()
            self.file.write("after iteration: " + str(multi_count[0]) + "\n")

        result = multi_dissagregator.get_statts()
        self.file.write("amount of multi states: " + str(result["count_multi"]) + "\n")
        self.file.write("correct: " + str(result["correct"]) + "\n")
        self.file.write("total: " + str(result["total"]) + "\n")
        self.file.write("correct multi: " + str(result["correct_multi"]) + "\n")
        self.file.write("total multi: " + str(result["total_multi"]) + "\n")
        self.file.write("accuracy multi states only: " + result["accuracy_multi"] + "\n")
        self.file.write("accuracy total: " + result["accuracy"] + "\n")
        self.file.write("\n")

        self.file.write("multi appliance dis\n")
        accuracy_mul = result["accuracy_multi"]
        caption = "of the" + (
            " improved " if self.is_improved else " ") + "multi appliance disaggregator " + \
                  ("" if breakpoints is None else "custom") + " using " + self.dataname
        label = "mul" + ("_improved_" if self.is_improved else "_") + self.test_data + (
            "" if breakpoints is None else "_custom")
        self.file.write(
            multi_to_latex(result["per_appliance"], self.order_appliances, caption, label, accuracy_mul))
        self.file.write("\n\n")

    def principal_component_analysis(self):
        self.file.write("\n\n PCA\n")
        print("PCA")
        pca = PCA()
        x2 = self.signal.get_input_sl()
        pca.fit(x2)
        self.file.write(str(pca.explained_variance_ratio_) + "\n")
        self.file.write(str(pca.singular_values_) + "\n")

    def weight_analysis(self):
        print("\n\nWEIGHT ANALYSIS\n")
        weights = np.absolute(self.label_clf.coefs_[0])
        x2 = np.array(self.signal.get_input_sl())
        x2_average = x2.mean(axis=0)
        x2_max = x2.max(axis=0)
        weights_average = weights.mean(axis=1)
        importance = weights_average * x2_average
        self.file.write("average\n")
        self.file.write(str(importance) + "\n")
        importance_max = weights_average * x2_max
        self.file.write(str(importance_max) + "\n")
        plt.bar([1, 2, 3, 4, 5, 6, 7, 8, 9], importance, color=(0.9, 0.4, 0.3),
                tick_label=["avg", "min", "shape", "dur", "hour", "prev", "temp", "wind", "weather"])
        plt.show()

    def scratch(self):
        print(len(np.where(np.array(self.signal.get_labels()) == 11)[0]))
        (unique, counts) = np.unique(self.signal.get_labels(), return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        self.file.write(str(frequencies) + "\n")


def main():
    """ you may set IMPROVED to either True or False, depending on if you want to test the improved version or not
        you may set TEST_DATA to either 'REDD' or 'STUDIO' depening on if you want to test
        on the REDD dataset or the STUDIO dataset"""

    schedule = [{"test_data": "REDD", "is_improved": False},
                {"test_data": "STUDIO", "is_improved": False},
                {"test_data": "REDD", "is_improved": True},
                {"test_data": "STUDIO", "is_improved": True},
                {"test_data": "GEN", "is_improved": True},
                {"test_data": "GEN", "is_improved": False}
                ]

    for batch in schedule:
        tester = Tester(batch["test_data"], batch["is_improved"])
        # tester.test_breakpoint_identifier()
        # tester.test_segment_labeler()
        tester.test_multi_appliance_dissagregator()
        tester.test_multi_appliance_dissagregator_custom()
        # tester.principal_component_analysis()
        # tester.weight_analysis()


if __name__ == "__main__":
    main()
