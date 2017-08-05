# encoding: utf-8

'''

@author: Soofo

@license: (C) Copyright 2017-2018, IECAS Limited.

@contact: iecaser@163.com

@software: pycharm

@file: ampli.py

@time: 17-7-29 下午4:41

@desc:

'''

submitfile = '/media/soffo/MEDIA/WORKSPACE/competition/tc/jiaotong/data/mydata/randsubmit/submit3/sb.txt'
import pandas as pd
import numpy as np

s = pd.read_csv(submitfile, sep='#')
s['travel_time'] *= 0.8
s['travel_time'] += np.random.randn(s.shape[0])
s.to_csv('/media/soffo/MEDIA/WORKSPACE/competition/tc/jiaotong/data/mydata/randsubmit/submit3/ampli.txt', sep='#',
         index=False)
