import csv
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from nilmtk import DataSet


class Datareader:
    @staticmethod
    def get_temperature_index(date):
        startDate = datetime(2011, 4, 17)
        deltadate = date - startDate
        index = ((deltadate.total_seconds() / 60) / 60) / 6
        return round(index)

    @staticmethod
    def load_temperature_data(filename):
        result = []
        dates = []
        with open("../data/" + filename, "r", encoding="utf8", newline='') as csvfile:
            file = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i, row in enumerate(file):
                if i > 0:
                    dates.append(datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
                    result.append([int(row[1]), int(row[2]), int(row[3]), int(row[4])])

        return pd.DataFrame(result, index=dates)

    @staticmethod
    def load_own_power_usage_data(name, sample_period):
        assert sample_period != 10
        sample_period = int(sample_period / 10)
        if not name.endswith(".csv"):
            print("can only read csv files")
            name = name + ".csv"

        devices = None
        power = []
        dates = []

        with open(name, "r", encoding="utf8", newline='') as csvfile:
            file = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            print("starting reading")

            for i, row in enumerate(file):
                print("iteration " + str(i) + " : " + ("%.2f%%" % ((i / 181440) * 100)), end='\r')
                if i == 0:
                    devices = row[1:]
                else:
                    dates.append(datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
                    power.append(np.array(row[1:]).astype(np.float))

        result = pd.DataFrame(power, columns=devices, index=dates).ffill(axis=0)

        print("sampling data")
        sampled_result = []
        sampled_result_index = []
        for i in range(0, len(result), sample_period):
            print("sample " + str(int(i/sample_period)) + " : " + ("%.2f%%" % ((i / 181440) * 100)), end='\r')
            sampled_result.append(result.iloc[i:sample_period+i].mean(axis=0))
            sampled_result_index.append(result.index[i])

        sampled_result = pd.DataFrame(sampled_result,sampled_result_index)

        print("\ndone reading")
        return sampled_result

    @staticmethod
    def load_appliances(elec_meter, selection, sample_period=3):
        result = []
        for s in selection:
            result.append(next(elec_meter[s].load(sample_period=sample_period)).ffill(axis=0))
        return result

    @staticmethod
    def load_appliances_selection(elec_meter, order, selection, sample_period=3):
        result = []
        dummy_data = next(elec_meter[3].load(sample_period=sample_period)).ffill(axis=0)
        length = len(dummy_data)
        row_names = dummy_data.index
        dummy_data = None
        for s in order:
            if s in selection:
                data = next(elec_meter[s].load(sample_period=sample_period)).ffill(axis=0)
                print(str(len(data)) + " + " + str(length))
                assert len(data) == length
                result.append(data)
                print(s + " has been loaded")
            else:
                result.append(pd.DataFrame(np.zeros(length), index=row_names))
                print(s + " has been replaced with 0's")

        return result

    @staticmethod
    def load_formatted_appliances(dataset: DataSet, start_date: datetime, end_date: datetime, building: int,
                                  selection: [object],
                                  activation_series_parameters: [[int]],
                                  sample_period: int = 60):

        dataset.set_window(start=start_date.strftime("%Y-%m-%d %H:%M:%S"), end=end_date.strftime("%Y-%m-%d %H:%M:%S"))
        elec_meter = dataset.buildings[building].elec
        idx = pd.date_range(start_date, end_date - timedelta(seconds=sample_period), freq=str(sample_period) + 'S')
        result = []
        total = None
        for i, s in enumerate(selection):
            min_off_duration = activation_series_parameters[i][0]
            min_on_duration = activation_series_parameters[i][1]
            on_power_threshold = activation_series_parameters[i][2]
            value = elec_meter[s].activation_series(min_off_duration=min_off_duration,
                                                    min_on_duration=min_on_duration,
                                                    on_power_threshold=on_power_threshold,
                                                    sample_period=sample_period,
                                                    border=0)
            data = pd.concat(value)
            data = data.reindex(idx, fill_value=0)
            result.append(data)
            if total is None:
                total = data
            else:
                total = total.add(data)

        return result, total

    @staticmethod
    def load_appliance_activation(dataset: DataSet, start_date: datetime, end_date: datetime, building: int,
                                  selection: [object],
                                  activation_series_parameters: [[int]],
                                  sample_period: int = 60):

        dataset.set_window(start=start_date.strftime("%Y-%m-%d %H:%M:%S"), end=end_date.strftime("%Y-%m-%d %H:%M:%S"))
        elec_meter = dataset.buildings[building].elec
        result = []
        for i, s in enumerate(selection):
            min_off_duration = activation_series_parameters[i]["min_off"]
            min_on_duration = activation_series_parameters[i]["min_on"]
            on_power_threshold = activation_series_parameters[i]["on_power_threshold"]
            value = elec_meter[s].activation_series(min_off_duration=min_off_duration,
                                                    min_on_duration=min_on_duration,
                                                    on_power_threshold=on_power_threshold,
                                                    sample_period=sample_period,
                                                    border=0)
            result.append(value)

        return result

    @staticmethod
    def load_total_power_consumption(elec_meter, selection, sample_period=3):
        return next(elec_meter.select_using_appliances(
            type=selection).load(sample_period=sample_period)).ffill(axis=0)
