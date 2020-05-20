from datetime import datetime
import pytz

SAMPLE_PERIOD = 60
order_appliances = ['fridge', 'microwave', 'washer dryer',
                    'dish washer']
breakpoint_classification = [
    {"max_power": 300, "on_power_threshold": 100, "min_on": int(60 / SAMPLE_PERIOD),
     "min_off": int(12 / SAMPLE_PERIOD)},
    {"max_power": 3000, "on_power_threshold": 200, "min_on": int(12 / SAMPLE_PERIOD),
     "min_off": int(30 / SAMPLE_PERIOD)},
    {"max_power": 2500, "on_power_threshold": 100, "min_on": int(1800 / SAMPLE_PERIOD),
     "min_off": int(160 / SAMPLE_PERIOD)},
    {"max_power": 2500, "on_power_threshold": 100, "min_on": int(1800 / SAMPLE_PERIOD),
     "min_off": int(600 / SAMPLE_PERIOD)}
]
selection_of_houses = {1: ["fridge", "microwave", 'washer dryer', 'dish washer'],  #
                       2: ["fridge", "microwave"],
                       3: ["fridge", "microwave", 'washer dryer'],
                       4: ['washer dryer']}
datetime_format = "%Y-%m-%d %H:%M:%S"
window_selection_of_houses = {1: (datetime(2011, 4, 18, 10, tzinfo=pytz.UTC).strftime(datetime_format),
                                  datetime(2011, 5, 1, tzinfo=pytz.UTC).strftime(datetime_format)),
                              2: (None, datetime(2011, 5, 1, tzinfo=pytz.UTC).strftime(datetime_format)),
                              3: (None, datetime(2011, 4, 27, tzinfo=pytz.UTC).strftime(datetime_format)),
                              4: (None, datetime(2011, 5, 1, tzinfo=pytz.UTC).strftime(datetime_format))}