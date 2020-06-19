from nilmtk import DataSet
import numpy as np

from datareader import Datareader as dr
from constants import breakpoint_classification
#TODO remove breakpoint classification
from signals import Signals
from dataprepairer import Dataprepairer as dp


class REDDloader:

    def __init__(self, window_selection: {}, appliance_selection: {}, order_appliances: [],
                 sample_rate: int, improved: bool):
        self.dataset = DataSet("../data/redd.5h")
        self.window_selection = window_selection
        self.appliance_selection = appliance_selection
        self.order_appliances = order_appliances
        self.sample_rate = sample_rate
        self.improved = improved

    def load_house(self, house: int):
        print("loading house: " + str(house))
        selection = self.appliance_selection[house]
        window_start, window_end = self.window_selection[house]
        self.dataset.set_window(start=window_start, end=window_end)
        elec = self.dataset.buildings[house].elec

        train_appliances = dr.load_appliances_selection(elec, self.order_appliances, selection,
                                                        self.sample_rate)
        train_total = dr.load_total_power_consumption(elec, selection, self.sample_rate)
        signals = Signals(self.sample_rate, self.order_appliances, breakpoint_classification,
                          self.improved, "temperature_redd.csv")
        signals.set_signals(train_appliances, train_total)

        return signals

    def concat_houses(self, houses_list, include_fake_breakpoint=False):

        _x1, _y1, _x2, _y2 = None, None, None, None

        for i in range(0, len(houses_list)):
            house = houses_list[i]
            signals = self.load_house(house)

            x1_part, y1_part, x2_part, y2_part = signals.get_prepared_data()

            if include_fake_breakpoint:
                x2_fake, y2_fake = dp.create_fake_breakpoints(signals)
                x2_part = np.concatenate((x2_part, x2_fake))
                y2_part = np.concatenate((y2_part, y2_fake))

            _x1 = x1_part if _x1 is None else np.concatenate((_x1, x1_part))
            _y1 = y1_part if _y1 is None else np.concatenate((_y1, y1_part))
            _x2 = x2_part if _x2 is None else np.concatenate((_x2, x2_part))
            _y2 = y2_part if _y2 is None else np.concatenate((_y2, y2_part))

        return _x1, _y1, _x2, _y2
