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


class Trainer:

    def __init__(self, test_data: str, is_improved: bool, do_cross: bool, print_report: bool = False):
        rcParams['figure.figsize'] = (13, 6)
        assert test_data == "REDD" or test_data == "STUDIO" or test_data == "GEN"
        assert (test_data == "GEN" and not do_cross) or not test_data == "GEN"
        self.test_data = test_data
        self.is_improved = is_improved
        self.do_cross = do_cross
        self.print_report = print_report
        self.sample_period = constants.SAMPLE_PERIOD[self.test_data]
        (self.x1_train, self.y1_train, self.x2_train, self.y2_train), (
            self.x1_test, self.y1_test, self.x2_test, self.y2_test), (
            self.x1, self.y1, self.x2, self.y2) = self.load_data()

    def start(self):
        if self.do_cross:
            bi_score, sl_score = self.cross()
            self.save_cross_score(bi_score, sl_score)

        bi_model, sl_model = self.train()
        self.save_models(bi_model, sl_model)

    def load_data(self):
        if self.test_data == "REDD":
            return self.load_redd()
        elif self.test_data == "STUDIO":
            return self.load_studio()
        else:
            return self.load_gen()

    def load_redd(self):
        print("############################")
        print("####### loading REDD #######")
        print("############################")
        print("##### improved = " + str(self.is_improved) + " ######")
        print("############################")
        houses = [1, 2, 3, 4]
        selection_of_appliances = constants.selection_of_appliances
        order_appliances = constants.order_appliances

        x1_train, y1_train, x2_train, y2_train = REDDloader(constants.window_selection_of_houses,
                                                            selection_of_appliances, order_appliances,
                                                            self.sample_period,
                                                            self.is_improved).concat_houses(houses, self.is_improved)
        x1_test, y1_test, x2_test, y2_test = REDDloader(constants.window_selection_of_houses_test,
                                                        selection_of_appliances, order_appliances, self.sample_period,
                                                        self.is_improved).concat_houses([1])
        x1, y1, x2, y2 = REDDloader(constants.window_selection_of_houses_complete,
                                    selection_of_appliances, order_appliances, self.sample_period,
                                    self.is_improved).concat_houses(houses)

        return (x1_train, y1_train, x2_train, y2_train), (x1_test, y1_test, x2_test, y2_test), (x1, y1, x2, y2)

    def load_studio(self):
        print("############################")
        print("###### loading STUDIO ######")
        print("############################")
        print("############################")
        print("##### improved = " + str(self.is_improved) + " ######")
        print("############################")
        appliances = dr.load_own_power_usage_data("../data/studio_data.csv", self.sample_period)
        split = int((len(appliances) * 3) / 4)
        x1_train, y1_train, x2_train, y2_train = STUDIOloader(self.sample_period, self.is_improved,
                                                              appliances=appliances, split=(None, split)).load(
            self.is_improved)
        x1_test, y1_test, x2_test, y2_test = STUDIOloader(self.sample_period, self.is_improved, appliances=appliances,
                                                          split=(split, None)).load()
        x1, y1, x2, y2 = STUDIOloader(self.sample_period, self.is_improved, appliances=appliances).load()

        return (x1_train, y1_train, x2_train, y2_train), (x1_test, y1_test, x2_test, y2_test), (x1, y1, x2, y2)

    def load_gen(self):
        print("############################")
        print("# loading REDD and STUDIO  #")
        print("############################")
        print("############################")
        print("##### improved = " + str(self.is_improved) + " ######")
        print("############################")
        houses = [1, 2, 3, 4]
        x1_train, y1_train, x2_train, y2_train = REDDloader(constants.window_selection_of_houses_complete,
                                                            constants.selection_of_generalizable_appliances,
                                                            constants.order_appliances_gen_REDD, self.sample_period,
                                                            self.is_improved).concat_houses(houses, self.is_improved)
        x1_test, y1_test, x2_test, y2_test = STUDIOloader(self.sample_period, self.is_improved,
                                                          order_appliances=constants.order_appliances_gen_STUDIO).load()

        return (x1_train, y1_train, x2_train, y2_train), (x1_test, y1_test, x2_test, y2_test), (None, None, None, None)

    def cross(self):
        breakpoint_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu", learning_rate="adaptive")
        label_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu", learning_rate="adaptive")
        print("calculating 10-fold cross validation score")
        scores_breakpoints = cross_val_score(breakpoint_clf, self.x1, self.y1, cv=10)
        print("Accuracy breakpoints: %0.2f (+/- %0.2f)" % (scores_breakpoints.mean(), scores_breakpoints.std() * 2))
        scores_label = cross_val_score(label_clf, self.x2, self.y2, cv=10)
        print("Accuracy label: %0.2f (+/- %0.2f)" % (scores_label.mean(), scores_label.std() * 2))
        return scores_breakpoints, scores_label

    def train(self):
        best_bi_model, best_sl_model = None, None
        best_accuracy_bi, best_accuracy_sl = 0, 0

        for i in range(0, 20):
            breakpoint_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu",
                                           learning_rate="adaptive")
            label_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu", learning_rate="adaptive")

            ros = RandomOverSampler(random_state=0)
            x1_resampled, y1_resampled = ros.fit_resample(self.x1_train, self.y1_train)
            print("training breakpoint identifier on iteration: " + str(i))
            breakpoint_clf.fit(x1_resampled, y1_resampled)
            x2_resampled, y2_resampled = ros.fit_resample(self.x2_train, self.y2_train)
            print("training segment labeler on iteration: " + str(i))
            label_clf.fit(x2_resampled, y2_resampled)

            pred_y1 = breakpoint_clf.predict(self.x1_test)
            pred_y2 = label_clf.predict(self.x2_test)

            if self.print_report:
                print("breakpoint classifier")
                print(classification_report(self.y1_test, pred_y1))
                print(confusion_matrix(self.y1_test, pred_y1))
            accuracy_bi = accuracy_score(self.y1_test, pred_y1)
            print("accuracy bi: " + str(accuracy_bi))

            if self.print_report:
                print("segment labeler")
                print(classification_report(self.y2_test, pred_y2))
                print(confusion_matrix(self.y2_test, pred_y2))
            accuracy_sl = accuracy_score(self.y2_test, pred_y2)
            print("accuracy sl: " + str(accuracy_sl))

            if accuracy_bi > best_accuracy_bi:
                best_accuracy_bi = accuracy_bi
                best_bi_model = breakpoint_clf

            if accuracy_sl > best_accuracy_sl:
                best_accuracy_sl = accuracy_sl
                best_sl_model = label_clf

        print("best accuracy bi: %0.2f" % best_accuracy_bi)
        print("best accuracy sl: %0.2f" % best_accuracy_sl)
        return best_bi_model, best_sl_model

    def save_models(self, bi_model, sl_model):
        path = self.get_path()
        dump(bi_model, path + 'breakpointidentifier2.ml')
        dump(sl_model, path + 'segmentlabeler2.ml')

    def save_cross_score(self, bi_score, sl_score):
        path = self.get_path()
        with open(path + "cross_val_score.txt", "w") as file:
            file.write("Accuracy breakpoints: %0.2f (+/- %0.2f)" % (bi_score.mean(), bi_score.std() * 2))
            file.write("Accuracy label: %0.2f (+/- %0.2f)" % (sl_score.mean(), sl_score.std() * 2))

    def get_path(self):
        if self.is_improved:
            path = '../models/' + self.test_data + '/improved/'
        else:
            path = '../models/' + self.test_data + '/original/'

        return path


def main():
    print_report = False
    schedule = [# {"test_data": "REDD", "is_improved": False, "do_cross": True},
                # {"test_data": "STUDIO", "is_improved": False, "do_cross": True},
                # {"test_data": "REDD", "is_improved": True, "do_cross": True},
                # {"test_data": "STUDIO", "is_improved": True, "do_cross": True},
                {"test_data": "GEN", "is_improved": False, "do_cross": False},
                {"test_data": "GEN", "is_improved": True, "do_cross": False}]

    for batch in schedule:
        print("executing batch: " + str(batch))
        trainer = Trainer(test_data=batch["test_data"], is_improved=batch["is_improved"], do_cross=batch["do_cross"],
                          print_report=print_report)
        trainer.start()


if __name__ == "__main__":
    main()
