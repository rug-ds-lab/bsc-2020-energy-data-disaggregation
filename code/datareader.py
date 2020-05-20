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
    def load_temperature_data():
        result = []
        with open("../data/temperature.csv", "r", encoding="utf8", newline='') as csvfile:
            file = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i, row in enumerate(file):
                if i > 0:
                    date = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
                    result.append([date, int(row[1]), int(row[2]), int(row[3]), int(row[4])])

        return result

    @staticmethod
    def load_own_power_usage_data(name):
        devices = None
        power = []
        dates = []
        with open("../data/" + name + ".csv", "r", encoding="utf8", newline='') as csvfile:
            file = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            print("starting reading")

            for i, row in enumerate(file):
                if i == 0:
                    devices = row[1:]
                else:
                    dates.append(datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
                    power.append(np.array(row[1:]).astype(np.float))

        result = pd.DataFrame(power, columns=devices, index=dates)

        return result

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
                print(str(len(data))+" + "+str(length))
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
