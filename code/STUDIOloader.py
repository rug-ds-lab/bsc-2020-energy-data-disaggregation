from datetime import datetime

import numpy as np

from datareader import Datareader as dr
from signals import Signals
from constants import breakpoint_classification_my_data
from dataprepairer import Dataprepairer as dp


class STUDIOloader:
    def __init__(self, sample_rate: int, improved: bool, appliances=None,
                 window_selection: (datetime, datetime) = None, split: (int, int) = None, order_appliances: [] = None):
        self.sample_rate = sample_rate
        self.improved = improved

        if appliances is None:
            self.appliances = dr.load_own_power_usage_data("../data/studio_data.csv", sample_rate)
        else:
            self.appliances = appliances

        if order_appliances is None:
            self.order_appliances = list(self.appliances)
        else:
            self.order_appliances = order_appliances

        assert split is None or window_selection is None
        if not (window_selection is None):
            self.appliances = self.appliances.loc[window_selection[0]:window_selection[1]]
        if not (split is None):
            self.appliances = self.appliances.iloc[split[0]:split[1]]

    def get_signals(self):
        signals = Signals(self.sample_rate, self.order_appliances, breakpoint_classification_my_data,
                          self.improved, "temperature_mydata.csv")
        signals.set_signals_mydata(self.appliances)

        return signals

    def load(self, include_fake_breakpoint=False):

        signals = self.get_signals()
        # signals.save_stats("studio_data_characteristics.csv")

        _x1, _y1, _x2, _y2 = signals.get_prepared_data()

        if include_fake_breakpoint:
            x2_fake, y2_fake = dp.create_fake_breakpoints(signals)
            _x2 = np.concatenate((_x2, x2_fake))
            _y2 = np.concatenate((_y2, y2_fake))

        return _x1, _y1, _x2, _y2
