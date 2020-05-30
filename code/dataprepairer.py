from datetime import datetime
import numpy as np

from datareader import Datareader as dr
import pandas as pd


class Dataprepairer:

    @staticmethod
    def get_breaking_points(meter, br_class, mark=1):
        on_power_threshold = br_class["on_power_threshold"]
        on = False
        output = []
        values = np.array(meter.values).reshape(-1)
        for value in values:
            if (on and value < on_power_threshold) or (not on and value >= on_power_threshold):
                on = not on

            output.append(on)

        return np.array(output)

    @staticmethod
    def get_state(meter, br_class):
        on_power_threshold = br_class["on_power_threshold"]
        min_on = int(br_class["min_on"])
        min_off = int(br_class["min_off"])

        on_timer = 0
        off_timer = min_off
        on = False
        values = meter.values
        output = []
        prev_value = 0

        for _x, value in enumerate(values.reshape(-1)):
            on_timer += 1 if on else 0
            off_timer += 1 if not on else 0
            turned_off = on and value < on_power_threshold <= prev_value
            turned_on = not on and value >= on_power_threshold > prev_value
            legit_turned_off = turned_off and on_timer >= min_on and np.average(
                values[_x:_x + min_off]) / 2 < on_power_threshold
            legit_turned_on = turned_on and off_timer >= min_off and np.average(
                values[_x:_x + min_on]) / 2 > on_power_threshold

            if legit_turned_off or legit_turned_on:
                on = not on
                on_timer = 0
                off_timer = 0

            output.append(on)
            prev_value = value

        return np.array(output)

    @staticmethod
    def get_states(meters, br_class_list):
        result = None
        for meter, br_class in zip(meters, br_class_list):
            # state = Dataprepairer.get_state(meter, br_class)
            state = Dataprepairer.get_breaking_points(meter, br_class)
            state = state.reshape((len(state), 1))
            result = state if result is None else np.append(result, state, axis=1)

        return result

    # TODO: check this
    @staticmethod
    def states_to_breakpoints(states):
        prev_state = states[0]
        states_on_breakpoints = [prev_state]
        breakpoints = [1]
        for state in states[1:]:
            if not np.all(prev_state == state):
                states_on_breakpoints.append(state)
                breakpoints.append(1)
            else:
                breakpoints.append(0)
            prev_state = state

        return breakpoints, states_on_breakpoints

    @staticmethod
    def breakpoint_states_to_labels(states_on_breakpoints):
        result = []
        for state in states_on_breakpoints:
            labels = []
            for i, s in enumerate(state):
                if s:
                    labels.append(i + 1)
            result.append(labels)

        return result

    @staticmethod
    def parse_input_breakpoint_identifier(mains, breakpoints):
        result = []
        values = np.array(mains.values).reshape(-1)
        prev_value = values[0]
        total_power_since_bp = 0
        count_since_bp = 0
        minimum = float("inf")
        assert len(breakpoints) == len(values)
        for _i, value in enumerate(values):
            total_power_since_bp += value
            count_since_bp += 1
            mean = total_power_since_bp / count_since_bp
            delta = abs(value - prev_value)
            minimum = value if value < minimum else minimum
            prev_value = value
            # TODO: maybe add count_since_bp
            result.append([mean, delta])

            if breakpoints[_i] > 0:
                total_power_since_bp = 0
                count_since_bp = 0
                minimum = float("inf")

        return np.array(result)

    @staticmethod
    def parse_output_segment_labeling(appliances_states):
        result = []

        for state in appliances_states:
            if np.count_nonzero(state) == 0:
                result.append(0)
            elif np.count_nonzero(state) == 1:
                for i, s in enumerate(state):
                    if s:
                        result.append(i + 1)
                        break
            else:
                result.append(len(state) + 1)

        return result

    @staticmethod
    def get_segments(mains, breakpoints):
        segments = []
        last_point = 0
        assert len(mains.values) == len(breakpoints)
        for _i, brpoints in enumerate(breakpoints):
            if brpoints > 0 and _i > last_point:
                segments.append(mains.iloc[last_point:_i])
                last_point = _i
        segments.append(mains.iloc[last_point:len(breakpoints)])
        return segments

    @staticmethod
    def parse_input_segment_labeling(segments, labels):
        result = []

        assert len(segments) == len(labels)

        # calculate the average and min
        for _i, segment in enumerate(segments):
            values = np.array(segment.values).reshape(-1)
            average = values.mean()
            min_consumption = values.min()
            timedelta = segment.index[-1] - segment.index[0]
            duration = timedelta.days * 24 * 60 + timedelta.seconds / 60
            hours_of_day_start = segment.index[0].hour
            previous = labels[_i - 1] if _i > 0 else 0
            shape = 1 if min_consumption > average - 10 else 0
            result.append(
                [average, min_consumption, duration, hours_of_day_start, shape, previous])

        return result

    @staticmethod
    def parse_input_segment_labeling_improved(segments, labels, temperature_file):
        result = []
        weather_data = dr.load_temperature_data(temperature_file)
        assert len(segments) == len(labels)

        # calculate the average and min
        for _i, segment in enumerate(segments):
            values = np.array(segment.values).reshape(-1)

            average = values.mean()
            min_consumption = values.min()
            max = values.max()
            std = values.std() if -1000 < values.std() < 1000 else 0
            timedelta = segment.index[-1] - segment.index[0]
            duration = timedelta.days * 24 * 60 + timedelta.seconds / 60
            hours_of_day_start = segment.index[0].hour
            previous = labels[_i - 1] if _i > 0 else 0
            date = datetime.fromtimestamp(segment.index[0].timestamp())
            nearest_date = weather_data.index.get_loc(date, method='nearest')
            weather = weather_data.iloc[nearest_date]
            avg_temp = (weather[0] + weather[1]) / 2
            wind = weather[2]
            weather_state = weather[3]
            shape = 1 if min_consumption > average - 10 else 0
            result.append(
                [average, min_consumption, std, duration, hours_of_day_start, shape, previous, avg_temp, wind,
                 weather_state])

        return result

    @staticmethod
    def create_fake_breakpoints(signal):
        _y1 = signal.get_breakpoints()
        fake_breakpoints = np.zeros(len(_y1))
        br_count = np.count_nonzero(_y1)
        fake_breakpoints[:br_count] = 1
        np.random.shuffle(fake_breakpoints)
        x2_fake = signal.get_input_sl_custom(fake_breakpoints)
        y2_fake = signal.get_labels_custom(fake_breakpoints)
        return x2_fake, y2_fake

