import os
import sys
import nilmtk
from nilmtk import DataSet
from nilmtk.dataset_converters import convert_redd
from nilmtk.utils import print_dict, dict_to_html

if os.sep == "/":
    convert_redd("../data/low_freq", "../data/redd.5h")
else:
    convert_redd(r'C:\Users\job heersink\Desktop\Projects\AS-NALM\data\low_freq',r"C:\Users\job heersink\Desktop\Projects\AS-NALM\data\redd.5h")

