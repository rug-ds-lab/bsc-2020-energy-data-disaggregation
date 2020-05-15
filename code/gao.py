from datetime import datetime

import numpy as np
import pytz
from matplotlib import rcParams
from nilmtk import DataSet
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
# import tensorflow.compat.v1 as tf
# import tensorflow.keras as keras

from datareader import Datareader as dr
import pandas as pd
from joblib import dump, load

from imblearn.over_sampling import RandomOverSampler

"""
labels:
0: fridge
1: dish washer
2: light0
3: microwave
4: electric space heater
5: electric stove
6: light 1
7: light 2
8: electriv oven
9: washer dryer
"""


def get_breaking_points(meter, err, diff=10, mark=1):
    on = False
    values = meter.values
    output = []
    for value in values:
        if (on and value[0] < err - diff) or (not on and value[0] >= err + diff):
            on = not on
            output.append(mark)
        else:
            output.append(0)

    return output


def parse_input_breakpoint_identifier(mains, breakpoints):
    values = mains.values
    result = []
    prev_value = values[0][0]
    total_power_since_bp = 0
    count_since_bp = 0
    minimum = float("inf")
    assert len(breakpoints) == len(values)
    for _i, value in enumerate(values):
        total_power_since_bp += value[0]
        count_since_bp += 1
        mean = total_power_since_bp / count_since_bp
        delta = abs(value[0] - prev_value)
        minimum = value if value < minimum else minimum
        prev_value = value[0]
        result.append([mean, delta])

        if breakpoints[_i] > 0:
            total_power_since_bp = 0
            count_since_bp = 0
            minimum = float("inf")

    return np.array(result)


def get_appliance_breakpoints(appliances, err=100):
    result = []
    for appliance in appliances:
        result.append(get_breaking_points(appliance, err))

    return result


def parse_output_breakpoint_identifier(indv_breakpoints):
    result = np.zeros(len(indv_breakpoints[0]))
    for breakpoints in indv_breakpoints:
        
        assert len(breakpoints) == len(result)
        for j, value in enumerate(breakpoints):
            if value > 0:
                result[j] = 1

    return np.array(result)


# TODO: check what index holds what value
def parse_output_segment_labeling(appliances, breakpoints, err):
    result = []

    assert len(breakpoints) == len(appliances[0].values)

    for _i, br in enumerate(breakpoints):
        if br > 0 or _i == 0:
            thing = 0
            for j, appliance in enumerate(appliances):
                if appliance.values[_i] > err:
                    thing = j + 1 if thing == 0 else len(appliances) + 1

            result.append(thing)

    return result


def getSegments(mains, breakpoints):
    segments = []
    last_point = 0
    assert len(mains.values) == len(breakpoints)
    for _i, br in enumerate(breakpoints):
        if br > 0 and _i > last_point:
            segments.append(mains.iloc[last_point:_i])
            last_point = _i
    segments.append(mains.iloc[last_point:len(breakpoints)])
    return segments


def parse_input_segment_labeling(segments, labels):
    result = []
    print(segments[0].min())
    print(segments[0].max())
    print(segments[0].std())
    weather_data = dr.load_temperature_data()

    assert len(segments) == len(labels)

    # calculate the average and min
    for _i, segment in enumerate(segments):
        # TODO: performs slightly better with std
        average = segment.mean().values[0]
        min = segment.min().values[0]
        # max = segment.max().values[0]
        # std = segment.std().values[0] if -1000 < segment.std().values[0] < 1000 else 0
        timedelta = segment.index[-1] - segment.index[0]
        duration = timedelta.days * 24 * 60 + timedelta.seconds / 60
        hours_of_day_start = segment.index[0].hour
        previous = labels[_i - 1] if _i > 0 else 0
        date = datetime.fromtimestamp(segment.index[0].timestamp())
        weather = weather_data[dr.get_temperature_index(date)]
        # avg_temp = (weather[1] + weather[2]) / 2
        # wind = weather[3]
        # weather_state = weather[4]
        shape = 1 if min > average - 10 else 0
        result.append(
            [average, min, duration, hours_of_day_start, shape, previous])

    return result


def dissagregate_mutli_appliances(clf, clf_input, number_of_appliances):
    multi_label = number_of_appliances + 1
    result = clf.predict(clf_input)
    extra_labels = np.zeros(len(result))
    assert (not result[0] == multi_label) and (not result[len(result) - 1] == multi_label)

    for _i in range(1, len(result) - 1):
        if result[_i] == multi_label:
            left = _i - 1
            right = _i + 1
            while result[left] == multi_label and left > 0:
                left -= 1
            while result[right] == multi_label and right < len(result):
                right += 1

            assert (not result[left] == multi_label) and (not result[right] == multi_label)

            # we subtract the average power of the base signal from the multiple appliance segment
            average_avg = 0
            average_min = 0
            if clf_input[_i][0] > clf_input[left][0]:
                average_avg = clf_input[left][0]
                average_min = clf_input[left][1]

            if clf_input[_i][0] > clf_input[right][0]:
                average_avg += clf_input[right][0]
                average_min += clf_input[right][1]

            # we subtract both neighbors from the segment
            if average_avg == clf_input[right][0] + clf_input[left][0] and result[left] == result[right]:
                average_avg = average_avg / 2
                average_min = average_min / 2

            clf_input[_i][0] -= average_avg
            clf_input[_i][1] -= average_min if average_min < clf_input[_i][1] else clf_input[_i][1]
            print("average: " + str(average_avg) + " min: " + str(average_min))

            # TODO: change std as well

            extra_labels[_i] = clf.predict([clf_input[_i]])

    return extra_labels


# TODO: implement scaling
rcParams['figure.figsize'] = (13, 6)
train = DataSet("../data/redd.5h")
test = DataSet("../data/redd.5h")
building = 1
SAMPLE_PERIOD = 10

train_start = datetime(2011, 4, 18, 10, tzinfo=pytz.UTC)
train_end = datetime(2011, 5, 1, tzinfo=pytz.UTC)

test_start = datetime(2011, 5, 12, tzinfo=pytz.UTC)
test_end = datetime(2011, 5, 20, tzinfo=pytz.UTC)

train.set_window(start=train_start.strftime("%Y-%m-%d %H:%M:%S"), end=train_end.strftime("%Y-%m-%d %H:%M:%S"))
test.set_window(start=test_start.strftime("%Y-%m-%d %H:%M:%S"), end=test_end.strftime("%Y-%m-%d %H:%M:%S"))
train_elec = train.buildings[1].elec
test_elec = test.buildings[1].elec

# TODO: ariconditioner is not selected
order_appliances = ['fridge', 'microwave', 'washer dryer', 'dish washer']  # 'electric oven', 'electric stove'
# selected_total = ['fridge', 'dish washer', 'microwave', 'washer dryer']  # # 'electric oven', 'electric stove'

selected_h1 = ["fridge", "microwave", 'washer dryer', 'dish washer']  # TODO: dish washer shouldn't be in there
selected_h2 = ["fridge", "microwave"]
# selected_h3 = ["fridge", "microwave", 'washer dryer']  # there are two dryers
# TODO: house 3 and 6 is fucked
selected_h4 = ['washer dryer']
# selected_h6 = ["fridge", "air conditioner"]
selection_of_houses = [selected_h1, selected_h2, selected_h4]
houses = [1, 2, 4]

breakpoint_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu", learning_rate="adaptive",
                               max_iter=500)
label_clf = MLPClassifier(alpha=1e-6, hidden_layer_sizes=16, activation="relu", learning_rate="adaptive")

### get data information ###
temp_data_train = next(train_elec['fridge'].load(sample_period=SAMPLE_PERIOD)).ffill(axis=0)
temp_data_test = next(test_elec['fridge'].load(sample_period=SAMPLE_PERIOD)).ffill(axis=0)
length_test = len(temp_data_test)
length_train = len(temp_data_train)
row_names_train = temp_data_train.index
row_names_test = temp_data_test.index
temp_data_train = None
temp_data_test = None
# training
for i in range(0, 1):
    house = houses[i]
    print("training house: " + str(house))
    selection = selection_of_houses[i]
    train_elec = train.buildings[house].elec
    train_appliances = dr.load_appliances_selection(train_elec, order_appliances, selection, length_train,
                                                    row_names_train, SAMPLE_PERIOD)
    train_total = dr.load_total_power_consumption(train_elec, selection, SAMPLE_PERIOD)
    print(len(train_appliances[0]))
    print(len(train_total))
    assert len(train_appliances[0]) == len(train_total)

    train_appliance_breakpoints = get_appliance_breakpoints(train_appliances, 100)
    train_breakpoints = parse_output_breakpoint_identifier(train_appliance_breakpoints)
    train_input = parse_input_breakpoint_identifier(train_total, train_breakpoints)
    ros = RandomOverSampler(random_state=0)
    train_input_resampled, train_breakpoints_resampled = ros.fit_resample(train_input, train_breakpoints)

    # breakpoint_clf.partial_fit(train_input, train_breakpoints, classes=list(range(0, len(order_appliances) + 2)))
    # breakpoint_clf.fit(train_input, train_breakpoints)
    breakpoint_clf.fit(train_input_resampled, train_breakpoints_resampled)

    train_labels = parse_output_segment_labeling(train_appliances, train_breakpoints, 100)
    train_segments = getSegments(train_total, train_breakpoints)
    train_input_segment_label = parse_input_segment_labeling(train_segments, train_labels)

    # label_clf.partial_fit(train_input2, train_labels, classes=list(range(0, len(order_appliances) + 2)))
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(train_input_segment_label, train_labels)
    # label_clf.fit(train_input_segment_label, train_labels)
    label_clf.fit(X_resampled, y_resampled)

for i in range(0, 1):
    house = houses[i]
    selection = selection_of_houses[i]
    test_elec = test.buildings[house].elec
    test_appliances = dr.load_appliances_selection(test_elec, order_appliances, selection, length_test, row_names_test,
                                                   SAMPLE_PERIOD)
    test_total = dr.load_total_power_consumption(test_elec, selection, SAMPLE_PERIOD)

    test_appliance_breakpoints = get_appliance_breakpoints(test_appliances, 100)
    test_breakpoints = parse_output_breakpoint_identifier(test_appliance_breakpoints)

    test_input = parse_input_breakpoint_identifier(test_total, test_breakpoints)

    pred_clf_breakpoints = breakpoint_clf.predict(test_input)

    print("breakpoint classifier")
    print(classification_report(test_breakpoints, pred_clf_breakpoints))
    print(confusion_matrix(test_breakpoints, pred_clf_breakpoints))

    test_labels = parse_output_segment_labeling(test_appliances, test_breakpoints, 100)
    test_labels_after_clf = parse_output_segment_labeling(test_appliances, pred_clf_breakpoints, 100)

    test_segments = getSegments(test_total, test_breakpoints)
    test_segments_after_clf = getSegments(test_total, pred_clf_breakpoints)
    test_input2 = parse_input_segment_labeling(test_segments, test_labels)
    test_input2_after_clf = parse_input_segment_labeling(test_segments_after_clf, test_labels_after_clf)

    pred_label_clf = label_clf.predict(test_input2)
    pred_label_clf_after_breakpoint_prediction = label_clf.predict(test_input2_after_clf)

    print("segment labeler")
    print(classification_report(test_labels, pred_label_clf))
    print(confusion_matrix(test_labels, pred_label_clf))

    print("segment labeler using output from breakpoint classifier")
    print(classification_report(test_labels_after_clf, pred_label_clf_after_breakpoint_prediction))
    print(confusion_matrix(test_labels_after_clf, pred_label_clf_after_breakpoint_prediction))

    """scores_breakpoints = cross_val_score(breakpoint_clf, test_input, test_breakpoints, cv=10)
    print("Accuracy breakpoints: %0.2f (+/- %0.2f)" % (scores_breakpoints.mean(), scores_breakpoints.std() * 2))
    scores_labeler = cross_val_score(label_clf, test_input2, test_labels, cv=10)
    print("Accuracy labels: %0.2f (+/- %0.2f)" % (scores_labeler.mean(), scores_labeler.std() * 2))
    scores_labeler = cross_val_score(label_clf, test_input2_after_clf, test_labels_after_clf, cv=10)
    print("Accuracy combined: %0.2f (+/- %0.2f)" % (scores_labeler.mean(), scores_labeler.std() * 2))"""

    addedLabels = dissagregate_mutli_appliances(label_clf, test_input2, len(order_appliances))
    print("distinct labels:")
    print(np.unique(addedLabels))

    ### some staticstics ###
    test_appliance_breakpoints = np.array(test_appliance_breakpoints)
    print("Breakpoints per appliance:")
    print(np.count_nonzero(test_appliance_breakpoints, axis=1))
    print("Total Breakpoints:")
    print(np.count_nonzero(test_breakpoints))
    #TODO: total breakpoints is equal to the breakpoints of fridge??

# TODO use train_elec[5].get_activations()[1]

"""if its left
and right neighbor segments are from the same appliance, we
assume that the multiple appliance segment is lifted by a base
signal from the appliance. Therefore, we subtract the average
power of the base signal from the multiple appliance segment
and label the remainder using the labeling classifier."""

"""if the left and right neighbors of the
multi-appliance segment are from different appliances, we assume
that the multi-appliance segment contains contributions
from both neighbors. Therefore, we subtract both neighbors
from the segment and label the remainder."""

"""if a multi-appliance segment has other multiappliance
segments as neighbors, then we find the nearest
neighbors that are from a single appliance, S1 and S2, and
apply the same subtraction strategy as above to all multiappliance
segments between S1 and S2."""
