import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from matplotlib import pyplot as pp

path = r'fuzzy1hot_v3.csv'
dataset = pd.read_csv(path,sep=',',header=0,low_memory=False,infer_datetime_format=True)
keep_list=['T1_label','RH1_label','T2_label','RH2_label','T3_label','RH3_label','T4_label','RH4_label','T5_label','RH5_label','T6_label','RH6_label','T7_label','RH7_label','T8_label','RH8_label','T9_label','RH9_label','PRESSURE_label','WIND_label','VISIBILITY_label','APPLIANCE_label']
dataset[keep_list].to_csv(r'fuzzy_labels.csv')