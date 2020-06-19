import csv

import numpy as np
from dataprepairer import Dataprepairer as dp
import pandas as pd
import matplotlib.pyplot as plt


class Signals:

    def __init__(self, sample_rate, order_appliances, breakpoint_classifications, is_improved, weather_datafile=None):
        self.sample_rate = sample_rate
        self.order_appliances = order_appliances
        self.signals = None
        self.sum_signals = None
        self.bc = breakpoint_classifications
        self._reset_values()
        self.is_improved = is_improved
        assert (not is_improved) or (is_improved and not (weather_datafile is None))
        self.weather_datafile = weather_datafile

    def _reset_values(self):
        self.breakpoints = None
        self.labels = None
        self.single_labels = None
        self.states = None
        self.states_on_breakpoints = None
        self.input_bi = None
        self.input_sl = None
        self.segments = None

    def set_signals(self, signals, sum_signals):
        self._reset_values()
        self.signals = signals
        self.sum_signals = sum_signals

    def set_signals_mydata(self, signals):
        self._reset_values()
        self.signals = []

        for column_label in self.order_appliances:
            self.signals.append(signals[column_label])

        self.sum_signals = signals.sum(axis=1)

    def get_signals(self):
        assert self.signals
        return self.signals

    def get_sum_signals(self):
        assert self.signals
        return self.sum_signals

    def get_breakpoints(self):
        assert self.signals
        states = self.get_states()

        if self.breakpoints is None:
            self.breakpoints, self.states_on_breakpoints = dp.states_to_breakpoints(states)

        return self.breakpoints

    def get_states(self):
        assert self.signals
        if self.states is None:
            self.states = dp.get_states(self.signals, self.bc)

        return self.states

    def get_states_on_breakpoints(self):
        assert self.signals
        states = self.get_states()

        if self.states_on_breakpoints is None:
            self.breakpoints, self.states_on_breakpoints = dp.states_to_breakpoints(states)

        return self.states_on_breakpoints

    def get_labels(self):
        assert self.signals
        states_on_breakpoints = self.get_states_on_breakpoints()

        if self.labels is None:
            self.labels = dp.parse_output_segment_labeling(states_on_breakpoints)

        return self.labels

    def get_input_bi(self):
        assert self.signals
        breakpoints = self.get_breakpoints()

        if self.input_bi is None:
            self.input_bi = dp.parse_input_breakpoint_identifier(self.sum_signals, breakpoints)

        return self.input_bi

    def get_segments(self):
        assert self.signals
        breakpoints = self.get_breakpoints()

        if self.segments is None:
            self.segments = dp.get_segments(self.sum_signals, breakpoints)

        return self.segments

    def get_segments_custom(self, breakpoints):
        assert self.signals
        return dp.get_segments(self.sum_signals, breakpoints)

    def get_input_sl(self):
        assert self.signals
        labels = self.get_labels()
        segments = self.get_segments()

        if self.input_sl is None:
            if self.is_improved:
                self.input_sl = dp.parse_input_segment_labeling_improved(segments, labels, self.weather_datafile)
            else:
                self.input_sl = dp.parse_input_segment_labeling(segments, labels)

        return self.input_sl

    def get_states_on_breakpoints_custom(self, breakpoints):
        assert self.signals
        states = self.get_states()
        assert len(breakpoints) == len(states)
        states_on_breakpoints = [states[0]]

        for br, st in zip(breakpoints[1:], states[1:]):
            if br == 1:
                states_on_breakpoints.append(st)

        return states_on_breakpoints

    def get_labels_custom(self, breakpoints):
        assert self.signals
        states_on_breakpoints = self.get_states_on_breakpoints_custom(breakpoints)

        return dp.parse_output_segment_labeling(states_on_breakpoints)

    def get_input_sl_custom(self, breakpoints):
        assert self.signals
        labels = self.get_labels_custom(breakpoints)
        segments = dp.get_segments(self.sum_signals, breakpoints)

        if self.is_improved:
            return dp.parse_input_segment_labeling_improved(segments, labels, self.weather_datafile)
        else:
            return dp.parse_input_segment_labeling(segments, labels)

    def get_prepared_data(self):
        return self.get_input_bi(), self.get_breakpoints(), self.get_input_sl(), self.get_labels()

    def save_stats(self, file_name="data_stats.csv", plot=False, print_to_commandline=False):
        ap = self.get_signals()
        st_on_br_sum = 0
        for st_on_br in np.array(self.get_states_on_breakpoints()).T:
            st_on_br_sum += np.count_nonzero(st_on_br)

        with open("../data/data_characteristics/" + file_name, "w", encoding="utf8", newline='') as csvfile:
            file = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file.writerow(
                [file_name[:-4], "max", "mean on", "mean off", "mean on time", "activations",
                 "labels", "distribution"])
            for i, name in enumerate(self.order_appliances):
                consumption = ap[i]
                activations = self.get_states().T[i]
                activations_on_breakpoints = np.array(self.get_states_on_breakpoints()).T[i]
                labels = self.get_labels()
                not_activation = [not i for i in activations]
                max_power = ("%.2f" % consumption.max()) + " W"
                mean_power_on = ("%.2f" % consumption[activations].mean()) + " W"
                mean_power_off = ("%.2f" % consumption[not_activation].mean()) + " W"

                count = []
                counter = 0
                for act in activations:
                    if act:
                        counter += 1
                    elif counter > 0:
                        count.append(counter)
                        counter = 0

                mean_on = "%.2f" % (np.array(count).mean() * self.sample_rate)

                counter = 0
                prev = False
                for br in activations_on_breakpoints:
                    if (prev and not br) or (not prev and br):
                        counter += 1
                    prev = br
                nr_activations = counter
                nr_labels = np.count_nonzero(activations_on_breakpoints)
                distribution = "%.2f%%" % ((nr_labels / st_on_br_sum) * 100)

                if print_to_commandline:
                    print("\n### " + name + " ###")
                    print("max power consumption: " + str(max_power))
                    print("mean power consumption on : " + str(mean_power_on))
                    print("mean power consumption off : " + str(mean_power_off))
                    print("mean on time: " + str(mean_on))
                    print("nr of activations : " + str(nr_activations))
                    print("nr of labels: " + str(nr_labels))
                    print("distribution: " + distribution)

                file.writerow(
                    [name, max_power, mean_power_on, mean_power_off, mean_on, nr_activations,
                     nr_labels, distribution])

                if plot:
                    tempdata = pd.DataFrame(consumption)
                    tempdata["activations"] = activations * consumption.max()

                    tempdata.plot()
                    plt.title(name)
                    plt.show()