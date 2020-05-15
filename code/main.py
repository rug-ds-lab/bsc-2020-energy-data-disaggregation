from __future__ import print_function, division

import sys
import time
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from six import iteritems

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CO, FHMMExact, Hart85
import nilmtk.utils


def predict(clf, test_elec, sample_period, timezone):
    gt = {}

    # "ac_type" varies according to the dataset used.
    # Make sure to use the correct ac_type before using the default parameters in this code.

    # load the power from the mains in a chunk
    chunk = next(test_elec.mains().load(physical_quantity='power', ac_type='apparent', sample_period=sample_period))
    chunk_drop_na = chunk.dropna()

    #predict the chunk
    pred = clf.disaggregate_chunk(chunk_drop_na)

    for meter in test_elec.submeters().meters:
        #load the data from every meter
        gt[meter] = next(meter.load(physical_quantity='power', ac_type='active', sample_period=sample_period))

    # make a panda dataframe out of it
    gt = pd.DataFrame({k: v.squeeze() for k, v in iteritems(gt) if len(v)},
                             index=next(iter(gt.values())).index).dropna()

    # If everything can fit in memory
    gt_overall = gt
    pred_overall = pred

    # Having the same order of columns
    gt_overall = gt_overall[pred_overall.columns]

    # Intersection of index
    gt_index_utc = gt_overall.index.tz_convert("UTC")
    pred_index_utc = pred_overall.index.tz_convert("UTC")
    common_index_utc = gt_index_utc.intersection(pred_index_utc)

    common_index_local = common_index_utc.tz_convert(timezone)
    gt_overall = gt_overall.loc[common_index_local]
    pred_overall = pred_overall.loc[common_index_local]
    appliance_labels = [m for m in gt_overall.columns.values]
    gt_overall.columns = appliance_labels
    pred_overall.columns = appliance_labels
    return gt_overall, pred_overall


def main(argv):
    rcParams['figure.figsize'] = (13, 6)

    train = DataSet("../data/redd.5h")
    test = DataSet("../data/redd.5h")
    building = 1
    train.set_window(end="2011-04-30")
    test.set_window(start="2011-04-30")
    train_elec = train.buildings[building].elec
    test_elec = test.buildings[building].elec
    top_5_train_elec = train_elec.submeters().select_top_k(k=5)

    classifiers = {'CO': CO({}), 'FHMM': FHMMExact({})}
    predictions = {}
    sample_period = 120
    for clf_name, clf in classifiers.items():
        print("*" * 20)
        print(clf_name)
        print("*" * 20)
        start = time.time()

        clf.train(top_5_train_elec, sample_period=sample_period)
        end = time.time()
        print("Runtime =", end - start, "seconds.")
        gt, predictions[clf_name] = predict(clf, test_elec, sample_period, train.metadata['timezone'])
        appliance_labels = [m.label() for m in gt.columns.values]
        gt.columns = appliance_labels
        predictions[clf_name].columns = appliance_labels

    print(gt.head())
    print(predictions['CO'].head())
    print(predictions['FHMM'].head())




if __name__ == "__main__":
    main(sys.argv[1:])
