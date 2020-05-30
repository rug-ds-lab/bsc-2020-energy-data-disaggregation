from datetime import datetime
from random import choice

import numpy as np
import pytz
from matplotlib import rcParams
from sklearn.neural_network import MLPClassifier


from signals import Signals


class Multi_dissagregator:

    def __init__(self, signal: Signals, clf: MLPClassifier, order_appliances: [], sample_period: int, improved: bool,
                 breakpoints=None, labels=None):
        rcParams['figure.figsize'] = (13, 6)
        self.signal = signal
        self.sample_period = sample_period
        self.order_appliances = order_appliances
        self.improved = improved
        self.multi_appliance_label = len(order_appliances) + 1
        self.clf = clf

        if breakpoints is None:
            self.input_sl = signal.get_input_sl()
            self.labels = signal.get_labels()
            self.states_on_breakpoints = signal.get_states_on_breakpoints()
        else:
            self.input_sl = signal.get_input_sl_custom(breakpoints)
            if labels is None:
                self.labels = signal.get_labels_custom(breakpoints)
            else:
                self.labels = labels
            self.states_on_breakpoints = signal.get_states_on_breakpoints_custom(breakpoints)

        self.states_consumption = np.zeros((len(self.input_sl), len(order_appliances) + 2, 4))
        for i in range(0, len(self.input_sl)):
            label = self.labels[i]
            inp = self.input_sl[i]
            self.states_consumption[i][0][0] = 1
            self.states_consumption[i][label][0] = 1
            self.states_consumption[i][label][1] = inp[0]  # average
            self.states_consumption[i][label][2] = inp[1]  # min
            if improved:
                self.states_consumption[i][label][3] = inp[2]  # std

    def disaggregated_appliances(self, consumption_per_appliance):
        found_appliances = []
        not_found = []
        for j, ap in enumerate(consumption_per_appliance):
            if ap[0] == 1:
                found_appliances.append(j)
            else:
                not_found.append(j)
        return found_appliances, not_found

    def has_unkown_appliance(self, neighbour, this):
        this_found, this_not_found = self.disaggregated_appliances(this)
        neighbour_found, neighbour_not_found = self.disaggregated_appliances(neighbour)

        return np.any(np.isin(this_not_found, neighbour_found))

    def get_unkown_appliance(self, neighbour, this):
        this_found, this_not_found = self.disaggregated_appliances(this)
        neighbour_found, neighbour_not_found = self.disaggregated_appliances(neighbour)

        unfound = []
        for i, unkown in enumerate(np.isin(this_not_found, neighbour_found)):
            if unkown:
                unfound.append(this_not_found[i])

        return unfound

    def diss_signal(self):
        for i, consumption_per_appliance in enumerate(self.states_consumption):
            if consumption_per_appliance[self.multi_appliance_label][0] == 1:

                left = i - 1
                right = i + 1
                while left > 0 and (not self.has_unkown_appliance(self.states_consumption[left],
                                                                  consumption_per_appliance)):
                    left -= 1
                while right < len(self.states_consumption) and (
                not self.has_unkown_appliance(self.states_consumption[right],
                                              consumption_per_appliance)):
                    right += 1

                if left > 0 and right < len(self.states_consumption):
                    # we subtract the average power of the base signal from the multiple appliance segment
                    average_avg = 0
                    average_min = 0
                    average_std = 0
                    unkown_appliances_right = self.get_unkown_appliance(self.states_consumption[right],
                                                                        consumption_per_appliance)
                    unkown_appliances_left = self.get_unkown_appliance(self.states_consumption[left],
                                                                       consumption_per_appliance)

                    app_r = choice(unkown_appliances_right)
                    app_l = choice(unkown_appliances_left)
                    app_m = self.multi_appliance_label
                    app_r_cons = self.states_consumption[right][app_r]
                    app_l_cons = self.states_consumption[left][app_l]
                    app_m_cons = consumption_per_appliance[app_m]

                    assert app_r_cons[0] == 1 and app_l_cons[0] == 1
                    assert consumption_per_appliance[app_l][0] == 0 and consumption_per_appliance[app_r][0] == 0

                    if app_m_cons[1] > app_l_cons[1]:
                        average_avg += app_l_cons[1]
                        average_min += app_l_cons[2]
                        average_std += app_l_cons[3]
                        consumption_per_appliance[app_l][0] = 1
                        consumption_per_appliance[app_l][1] = app_l_cons[1]
                        consumption_per_appliance[app_l][2] = app_l_cons[2]
                        consumption_per_appliance[app_l][3] = app_l_cons[3]

                    if app_m_cons[1] > app_r_cons[1]:
                        average_avg += app_r_cons[1]
                        average_min += app_r_cons[2]
                        average_std += app_r_cons[3]
                        if not consumption_per_appliance[app_l][0] == 1:
                            consumption_per_appliance[app_r][0] = 1
                            consumption_per_appliance[app_r][1] = app_r_cons[1]
                            consumption_per_appliance[app_r][2] = app_r_cons[2]
                            consumption_per_appliance[app_r][3] = app_r_cons[3]

                    # we subtract both neighbors from the segment
                    if average_avg == app_r_cons[1] + app_l_cons[1] and app_r == app_l:
                        average_avg = average_avg / 2
                        average_min = average_min / 2
                        average_std = average_std / 2

                    consumption_per_appliance[app_m][1] -= average_avg
                    consumption_per_appliance[app_m][2] -= average_min
                    consumption_per_appliance[app_m][3] -= average_std

                    input_part = np.array(self.input_sl[i].copy())
                    input_part[0] = consumption_per_appliance[app_m][1]
                    input_part[1] = consumption_per_appliance[app_m][2]
                    if self.improved:
                        input_part[2] = consumption_per_appliance[app_m][3]

                    pred = self.clf.predict(input_part.reshape(1, -1))[0]
                    if not pred == self.multi_appliance_label:
                        consumption_per_appliance[pred][0] = 1
                        consumption_per_appliance[pred][1] = consumption_per_appliance[app_m][1]
                        consumption_per_appliance[pred][2] = consumption_per_appliance[app_m][2]
                        consumption_per_appliance[pred][3] = consumption_per_appliance[app_m][3]
                        consumption_per_appliance[app_m][0] = 0
                        consumption_per_appliance[app_m][1] = 0
                        consumption_per_appliance[app_m][2] = 0
                        consumption_per_appliance[app_m][3] = 0

    def count_multi_consumption_per_appliance(self):
        count = 0
        indexes = []
        for i, app_cons in enumerate(self.states_consumption):
            if app_cons[self.multi_appliance_label][0] == 1:
                count += 1
                indexes.append(i)

        return count, indexes

    def consumption_per_appliance_to_states(self):
        modified_state_on_breakpoints = []
        for app_cons in self.states_consumption:
            states = []
            for app in app_cons[1:len(app_cons) - 1]:
                states.append(app[0] == 1)

            modified_state_on_breakpoints.append(states)

        return modified_state_on_breakpoints

    def get_statts(self):
        true_values = np.array(self.states_on_breakpoints)
        predicted_values = np.array(self.consumption_per_appliance_to_states())

        correct = np.count_nonzero((true_values == predicted_values).reshape(-1))
        total = len((true_values == predicted_values).reshape(-1))
        multi_label_i = np.where(np.array(self.labels) == self.multi_appliance_label)[0]
        correct_multi = np.count_nonzero((true_values[multi_label_i] == predicted_values[multi_label_i]).reshape(-1))
        total_multi = len((true_values[multi_label_i] == predicted_values[multi_label_i]).reshape(-1))

        return correct, total, correct_multi, total_multi, len(multi_label_i)
