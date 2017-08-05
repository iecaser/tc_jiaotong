# encoding: utf-8

'''

@author: Soofo

@license: (C) Copyright 2017-2018, IECAS Limited.

@contact: iecaser@163.com

@software: pycharm

@file: lstmreg.py

@time: 2017/7/20 12:15

@desc:

'''

newtime = '[2016-06-%2d 08:00:00,2016-05-01 08:02:00)'
# 2016-05-01
import pandas as pd
import numpy as np
from tqdm import tqdm

workingPath = 'D:/WORKSPACE/competition/tc/jiaotong/'
data = pd.read_csv(workingPath + '/data/mydata/randsubmit/converted.csv', sep=';')
data = data.sort_values(by='time_interval')
data.index = [i for i in range(data.shape[0])]
print(sorted(list(set(data['time_interval']))))
oldTimeInterval = data['time_interval'].copy().values[:-30 * 132]
oldDate = data['date'].copy().values[:-30 * 132]
t1 = data[3 * 30 * 132:]
t2 = t1[:2 * 30 * 132]
data = pd.concat([t1, t2])
data['time_interval'] = oldTimeInterval
data['date'] = oldDate
# data = data[:-30 * 132]

# add random
r = np.random.randn(data.shape[0])
data['travel_time'] = data['travel_time'] + r
data.to_csv(workingPath + 'submitting.txt', sep='#', index=False, encoding='utf-8')
pass
