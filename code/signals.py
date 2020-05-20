import numpy as np
from dataprepairer import Dataprepairer as dp
import pandas as pd

class Signals:

    def __init__(self, sample_rate, label_to_appliance, breakpoint_classifications):
        self.sample_rate = sample_rate
        self.label_to_appliance = label_to_appliance
        self.signals = None
        self.sum_signals = None
        self.bc = breakpoint_classifications
        self._reset_values()

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

    def get_input_sl(self):
        assert self.signals
        labels = self.get_labels()
        segments = self.get_segments()

        if self.input_sl is None:
            self.input_sl = dp.parse_input_segment_labeling(segments, labels)

        return self.input_sl

