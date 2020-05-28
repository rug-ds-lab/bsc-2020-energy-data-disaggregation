import sys
from datetime import datetime

import numpy as np
import pytz
from matplotlib import rcParams
from nilmtk import DataSet


from datareader import Datareader as dr
from dataprepairer import Dataprepairer as dp
import pandas as pd

from signals import Signals
from constants import REDD_SAMPLE_PERIOD, order_appliances, breakpoint_classification, selection_of_houses, window_selection_of_houses
from tester import Tester

data = DataSet("../data/redd.5h")
houses = [1, 2, 3, 4]
for i in range(0, len(houses)):
    house = houses[i]
    print("statting house: " + str(house))
    selection = selection_of_houses[house]
    data_elec = data.buildings[house].elec

    train_appliances = dr.load_appliances_selection(data_elec, order_appliances, selection, REDD_SAMPLE_PERIOD)
    train_total = dr.load_total_power_consumption(data_elec, selection, REDD_SAMPLE_PERIOD)
    signals = Signals(REDD_SAMPLE_PERIOD, order_appliances, breakpoint_classification)
    signals.set_signals(train_appliances, train_total)
    signals.save_stats("REDD_char"+str(house)+".csv")


