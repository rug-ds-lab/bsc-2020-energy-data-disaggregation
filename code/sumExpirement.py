import csv
import sys
from datetime import datetime

import numpy as np
import pytz
from matplotlib import rcParams
from nilmtk import DataSet
from sklearn import preprocessing
import matplotlib.pyplot as plt

from datareader import Datareader as dr
from dataprepairer import Dataprepairer as dp
import pandas as pd

SAMPLE_PERIOD = 10
rcParams['figure.figsize'] = (12, 6)

appliances = dr.load_own_power_usage_data("experiment_data.csv", SAMPLE_PERIOD)
appliances.iloc[:500].plot()
plt.title("consumption ipad+lamp")
plt.ylabel("W")
plt.show()
appliances.plot()
plt.title("consumption ipad+lamp+water heater")
plt.ylabel("W")
plt.show()
comp = pd.DataFrame(appliances["total"])
comp["sum_appliances"] = appliances.iloc[:, 1:].sum(axis=1)
comp.plot()
plt.title("total vs summed appliances")
plt.ylabel("W")
plt.show()
diff = comp["total"]-comp["sum_appliances"]
diff.plot()
plt.title("difference")
plt.ylabel("W")
plt.show()

for i,d in enumerate(diff):
    if d < -200:
        print(i)
