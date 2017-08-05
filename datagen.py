# encoding: utf-8

'''

@author: Soofo

@license: (C) Copyright 2017-2018, IECAS Limited.

@contact: iecaser@163.com

@software: pycharm

@file: datagen.py

@time: 2017/7/20 12:15

@desc:

'''

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import numpy as np
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Conv1D
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint, EarlyStopping
import re
from tqdm import tqdm

# global
# windows
# workingPath = 'D:/WORKSPACE/competition/tc/jiaotong/'
# linux
workingPath = '/media/soffo/MEDIA/WORKSPACE/competition/tc/jiaotong/'
key0 = 'link_ID'
key1 = 'date'
key2 = 'time_interval'
key3 = 'travel_time'
# data generate
avgSize = 5
step = 60
featureSize = 132
predictTimes = 30
# predictTimes = 6
scaler = MinMaxScaler(feature_range=(0, 1))


def dataProcessing():
    # 注意read_csv中sep参数
    # raws = pd.read_csv(workingPath + 'data/gy_contest_link_traveltime_training_data.txt', sep=';')
    raws = pd.read_csv(workingPath + 'data/data.txt', sep=';')
    # raws = pd.read_csv(workingPath + 'data/minidata.txt', sep=';')
    # 注意重排序一定要index跟着变!! 否则后续逻辑将理不清
    print('sorting...')
    raws = raws.sort_values(by=[key2, key0])
    print('reindex...')
    raws.index = [i for i in range(raws.shape[0])]
    print('saving...')
    # raws.to_csv(workingPath + 'newraws.csv', index=False, sep=';')
    linkids = list(set(raws[key0]))
    timeIntervals = raws[key2]
    # uniqueTimeIntervals = sorted(list(set(timeIntervals)))
    # 补充数据
    moreLinkids, moreDates, moreTimeIntervals, moreTravelTime = [], [], [], []
    print('cutting...')
    # 数据太多,先只用6:00到9:00
    tf = np.zeros(raws.shape[0]).astype('bool')
    # [2016-05-21 06:00:00,2016-05-21 06:02:00)
    timePattern = '^\[2016-\d+-\d+ [0][678]'
    # timePattern = '^\[2016-05-\d+ 08'
    for i in tqdm(range(raws.shape[0])):
        tf[i] = (re.match(timePattern, timeIntervals[i]) is not None)
    raws = raws[tf]
    raws.to_csv(workingPath + 'data/minicut.csv', index=False, sep=';')
    # raws.to_csv(workingPath + 'cut.csv', index=False, sep=';')
    uniqueTimeIntervals = sorted(list(set(raws[key2])))
    print('avging...')
    for linkid in tqdm(linkids):
        # print(linkid)
        x = raws[raws[key0] == linkid]
        xTimeIntervals = x[key2]
        # x = x.sort_values(by='time_interval')
        for ti in uniqueTimeIntervals:
            if ti not in list(xTimeIntervals):
                roughIndex = raws[ti == timeIntervals].index[0]
                dist = np.abs(x.index - roughIndex)
                avg = 0
                for i in range(avgSize):
                    index = np.argsort(dist)[i]
                    avg += x.values[index][3]
                avg /= avgSize
                moreLinkids.append(linkid)
                moreDates.append(raws.loc[roughIndex].values[1])
                moreTimeIntervals.append(ti)
                moreTravelTime.append(avg)
                # print('-' * 100)
                # print(x.info())
                # print(x)
    moreinfo = pd.DataFrame(data={key0: moreLinkids, key1: moreDates, key2: moreTimeIntervals, key3: moreTravelTime})
    newindex = [raws.shape[0] + i for i in range(moreinfo.shape[0])]
    moreinfo.index = newindex
    raws = pd.concat([raws, moreinfo])
    # raws = raws.sort_values(by=[key2, key0])
    raws = raws.sort_values(by=[key0, key2])
    newindex = [i for i in range(raws.shape[0])]
    raws.index = newindex
    raws = raws[[key0, key1, key2, key3]]
    # 注意因为time_interval的str中间含有",",而csv以此为分割,故tocsv后,csv里面实际上这一段被加了双引号
    # loadcsv并不影响
    # raws.to_csv(workingPath + 'data/mydata/raws.csv', index=False, sep=';')
    raws.to_csv(workingPath + 'data/mydata/miniraws.csv', index=False, sep=';')
    # - old方式:raws为数据按照时间(首要排序)/id(次要排序)顺序排列
    # - new方式:raws为数据按照id(首要排序)/时间(次要排序)顺序排列,是为了reshape数据直接变为id为行,时刻为列的矩阵
    # - raws.shape[0]是132*720,其中132为linkid数目;720=24小时*60min/(2min为间隔)
    # ! 注意到有部分数据[严重]缺失,暂时用平均代替
    # ----------------------至此数据补充完整--------------------------------


# - 拟采用2个model进行预测
# - model1: lstm 特征完全来源于历史趋势
# - model2: 传统ML回归方法,特征无历史,包含length/width/时刻(时刻是出行的一个重要指标)
# - 后续将特征完全整合为一个model,以及ensemble


def dataGen():
    datapath = 'data/mydata/minitest/raws.csv'
    print('loading ' + datapath + ' ...')
    raws = pd.read_csv(workingPath + datapath, sep=';')
    dataset = raws[key3].values.reshape(featureSize, 720).transpose()
    dataset = scaler.fit_transform(dataset)
    # test data
    testx = dataset[-step:].reshape(1, step, -1)
    datax, datay = [], []
    # 以前采用step个数据预测1个数据,进而滑动下去,为直观在train时刻观察loss,改为step个数据预测下30个数据
    # predictTimes=1即为上述单点预测
    print('generating data...')
    for i in tqdm(range(dataset.shape[0] - step - predictTimes)):
        a = dataset[i:i + step]
        b = dataset[i + step:i + step + predictTimes]
        # 这里让b回归为按id->时刻排列的vector形式,输出重新做reshape即可
        b = b.transpose().flatten()
        datax.append(a)
        datay.append(b)
    datax = np.array(datax)
    datay = np.array(datay)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    trainx = datax[0:train_size, :]
    trainy = datay[0:train_size, :]
    valx = datax[train_size:len(dataset), :]
    valy = datay[train_size:len(dataset), :]
    print('data generated.')
    return trainx, trainy, valx, valy, testx


# ------------------------------- model1 ----------------------------------------
def getModel():
    model = Sequential()
    model.add(LSTM(featureSize * predictTimes, input_shape=(step, featureSize), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(featureSize * predictTimes))
    model.add(Dropout(0.2))
    model.add(Dense(featureSize * predictTimes))
    model.add(Dropout(0.5))
    model.add(Dense(featureSize * predictTimes))
    model.add(Dropout(0.5))
    model.add(Dense(featureSize * predictTimes))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# ------------ lstm train --------------------
def train(trainx, trainy, valx, valy, use_existing=True):
    print('-' * 50)
    print('getting model...')
    model = getModel()
    model_checkpoint = ModelCheckpoint(workingPath + 'code/lstm.hdf5', monitor='val_loss',
                                       save_best_only=True)
    if use_existing:
        print('loading weight...')
        model.load_weights(workingPath + 'code/lstm.hdf5')
    print('fitting...')
    model.fit(trainx, trainy, epochs=25, batch_size=2, verbose=1, callbacks=[model_checkpoint],
              validation_data=[valx, valy])
    return model


# ------------- predict --------------------------
def predict(testx):
    model = getModel()
    model.load_weights(workingPath + 'code/lstm.hdf5')
    predicty = model.predict(testx)
    return predicty


#
# dataProcessing()
trainx, trainy, valx, valy, testx = dataGen()
train(trainx, trainy, valx, valy, use_existing=False)
# train(trainx, trainy, valx, valy, use_existing=True)

# ------------------------------ 滑动预测:下面形成输出 ----------------------------------
# finaly = np.zeros((predictTimes, featureSize)).astype('float32')
# for i in range(90):
#     testy = predict(testx)
#     testx[0, :-1, :] = testx[0, 1:, :]
#     testx[0, -1] = testy
#     finaly[i] = testy

# rfinaly = scaler.inverse_transform(finaly)
# np.save(working_path + 'rfinaly.npy', rfinaly)
# priceLoc = np.arange(45) % 3 == 0
# # 90*15
# pricey = rfinaly[:, priceLoc]
# np.save(working_path + 'pricey.npy', pricey)
# pricey = np.load(working_path + 'pricey.npy')
# sprice = pricey.flatten()
# saddr = addrSort * 90
# sdata = pd.read_csv('my.csv')
# sdate = sorted(list(set(sdata['日期'])) * 15)
# s = pd.DataFrame({'日期': sdate, '地区': saddr, '价格': sprice})
# s = s[['日期', '地区', '价格']]
# s.to_csv(working_path + 's4.csv', encoding='utf-8', index=False)


pass
