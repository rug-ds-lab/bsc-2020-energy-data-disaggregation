from datetime import datetime
import pytz

REDD_SAMPLE_PERIOD = 60
STUDIO_SAMPLE_PERIOD = 10
order_appliances = ['fridge', 'microwave', 'washer dryer',
                    'dish washer']
order_appliances_gen_REDD = ['fridge', 'microwave', 'washer dryer']
order_appliances_gen_STUDIO = ['fridge', 'microwave', 'washing_machine']
breakpoint_classification = [
    {"max_power": 300, "on_power_threshold": 100, "min_on": int(60 / REDD_SAMPLE_PERIOD),
     "min_off": int(12 / REDD_SAMPLE_PERIOD)},
    {"max_power": 3000, "on_power_threshold": 200, "min_on": int(12 / REDD_SAMPLE_PERIOD),
     "min_off": int(30 / REDD_SAMPLE_PERIOD)},
    {"max_power": 2500, "on_power_threshold": 100, "min_on": int(1800 / REDD_SAMPLE_PERIOD),
     "min_off": int(160 / REDD_SAMPLE_PERIOD)},
    {"max_power": 2500, "on_power_threshold": 100, "min_on": int(1800 / REDD_SAMPLE_PERIOD),
     "min_off": int(600 / REDD_SAMPLE_PERIOD)}
]
breakpoint_classification_my_data = [
    # tv
    {"max_power": 65, "on_power_threshold": 15, "min_on": int(300 / STUDIO_SAMPLE_PERIOD),
     "min_off": int(300 / STUDIO_SAMPLE_PERIOD)},
    # phone_charger
    {"max_power": 26, "on_power_threshold": 5, "min_on": int(60 / STUDIO_SAMPLE_PERIOD),
     "min_off": int(60 / STUDIO_SAMPLE_PERIOD)},
    # desk_lamp
    {"max_power": 37, "on_power_threshold": 10, "min_on": int(10 / STUDIO_SAMPLE_PERIOD),
     "min_off": int(10 / STUDIO_SAMPLE_PERIOD)},
    # couch_lamp
    {"max_power": 54, "on_power_threshold": 10, "min_on": int(10 / STUDIO_SAMPLE_PERIOD),
     "min_off": int(10 / STUDIO_SAMPLE_PERIOD)},
    # washing_machine
    {"max_power": 2067, "on_power_threshold": 100, "min_on": int(3600 / STUDIO_SAMPLE_PERIOD),
     "min_off": int(1200 / STUDIO_SAMPLE_PERIOD)},
    # fridge
    {"max_power": 704, "on_power_threshold": 25, "min_on": int(30 / STUDIO_SAMPLE_PERIOD),
     "min_off": int(12 / STUDIO_SAMPLE_PERIOD)},
    # water_heater
    {"max_power": 1989, "on_power_threshold": 1000, "min_on": int(1 / STUDIO_SAMPLE_PERIOD),
     "min_off": int(1 / STUDIO_SAMPLE_PERIOD)},
    # alienware_laptop
    {"max_power": 144, "on_power_threshold": 20, "min_on": int(60 / STUDIO_SAMPLE_PERIOD),
     "min_off": int(60 / STUDIO_SAMPLE_PERIOD)},
    # ps4
    {"max_power": 152, "on_power_threshold": 60, "min_on": int(60 / STUDIO_SAMPLE_PERIOD),
     "min_off": int(60 / STUDIO_SAMPLE_PERIOD)},
    # microwave
    {"max_power": 2510, "on_power_threshold": 1000, "min_on": int(30 / STUDIO_SAMPLE_PERIOD),
     "min_off": int(30 / STUDIO_SAMPLE_PERIOD)}
]
selection_of_appliances = {1: ["fridge", "microwave", 'washer dryer', 'dish washer'],
                           2: ["fridge", "microwave"],
                           3: ["fridge", "microwave", 'washer dryer'],
                           4: ['washer dryer']}
selection_of_generalizable_appliances = {1: ["fridge", "microwave", 'washer dryer'],
                                         2: ["fridge", "microwave"],
                                         3: ["fridge", "microwave", 'washer dryer'],
                                         4: ['washer dryer']}
datetime_format = "%Y-%m-%d %H:%M:%S"
window_selection_of_houses = {1: (datetime(2011, 4, 18, 10, tzinfo=pytz.UTC).strftime(datetime_format),
                                  datetime(2011, 5, 1, tzinfo=pytz.UTC).strftime(datetime_format)),
                              2: (None, datetime(2011, 5, 1, tzinfo=pytz.UTC).strftime(datetime_format)),
                              3: (None, datetime(2011, 4, 27, tzinfo=pytz.UTC).strftime(datetime_format)),
                              4: (None, datetime(2011, 5, 1, tzinfo=pytz.UTC).strftime(datetime_format))}

window_selection_of_houses_test = {1: (datetime(2011, 5, 12, tzinfo=pytz.UTC).strftime(datetime_format),
                                       datetime(2011, 5, 20, tzinfo=pytz.UTC).strftime(datetime_format))}

window_selection_of_houses_complete = {1: (datetime(2011, 4, 18, 10, tzinfo=pytz.UTC).strftime(datetime_format),
                                           datetime(2011, 5, 20, tzinfo=pytz.UTC).strftime(datetime_format)),
                                       2: (None, datetime(2011, 5, 1, tzinfo=pytz.UTC).strftime(datetime_format)),
                                       3: (None, datetime(2011, 4, 27, tzinfo=pytz.UTC).strftime(datetime_format)),
                                       4: (None, datetime(2011, 5, 1, tzinfo=pytz.UTC).strftime(datetime_format))}
