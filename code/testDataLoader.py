from datetime import datetime

import pytz
from nilmtk import DataSet

from signals import Signals
import constants
from datareader import Datareader as dr


class TestDataLoader:

    @staticmethod
    def load_REDD(is_improved):
        test = DataSet("../data/redd.5h")
        start = datetime(2011, 5, 12, tzinfo=pytz.UTC)
        end = datetime(2011, 5, 20, tzinfo=pytz.UTC)
        test.set_window(start=start.strftime("%Y-%m-%d %H:%M:%S"), end=end.strftime("%Y-%m-%d %H:%M:%S"))
        test_elec = test.buildings[1].elec
        selection = constants.selection_of_houses[1]
        test_appliances = dr.load_appliances_selection(test_elec, constants.order_appliances, selection,
                                                       constants.REDD_SAMPLE_PERIOD)
        test_total = dr.load_total_power_consumption(test_elec, selection, constants.REDD_SAMPLE_PERIOD)
        test_signals = Signals(constants.REDD_SAMPLE_PERIOD, constants.order_appliances,
                               constants.breakpoint_classification, is_improved, "temperature_redd.csv")
        test_signals.set_signals(test_appliances, test_total)

        return test_signals, constants.order_appliances, constants.REDD_SAMPLE_PERIOD

    @staticmethod
    def load_STUDIO(is_improved):
        appliances = dr.load_own_power_usage_data("studio_data.csv", constants.STUDIO_SAMPLE_PERIOD)
        order_appliances = list(appliances)
        test_signals = Signals(constants.STUDIO_SAMPLE_PERIOD, order_appliances,
                               constants.breakpoint_classification_my_data, is_improved,
                               "temperature_mydata.csv")
        split = int((len(appliances) * 3) / 4)
        test_signals.set_signals_mydata(appliances.iloc[split:])

        return test_signals, order_appliances, constants.STUDIO_SAMPLE_PERIOD
