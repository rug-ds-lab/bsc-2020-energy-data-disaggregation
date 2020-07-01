from datareader import Datareader as dr
import pandas as pd
import numpy as np


def print_stats(temp: pd.DataFrame):
    avg_temp = (temp[0].mean() + temp[1].mean()) / 2
    std_temp = (temp[0].std() + temp[1].std()) / 2
    avg_wind = temp[2].mean()
    std_wind = temp[2].std()
    nr_occurences = len(temp[3])
    rain = np.count_nonzero(temp[3] == 0) / nr_occurences * 100
    cloud = np.count_nonzero(temp[3] == 1) / nr_occurences * 100
    mist = np.count_nonzero(temp[3] == 2) / nr_occurences * 100
    sunny = np.count_nonzero(temp[3] == 3) / nr_occurences * 100
    print("avg temp: " + str(avg_temp) + " std: " + str(std_temp))
    print("avg wind: " + str(avg_wind) + " std: " + str(std_wind))
    print("r: " + str(rain) + "c: " + str(cloud) + "m: " + str(mist) + "s: " + str(sunny))


temp_REDD = dr.load_temperature_data("temperature_redd.csv")
temp_studio = dr.load_temperature_data("temperature_mydata.csv")
print("REDD")
print_stats(temp_REDD)
print("STUDIO")
print_stats(temp_studio)