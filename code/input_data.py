# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import numpy

from six.moves import urllib
from six.moves import xrange	# pylint: disable=redefined-builtin
import tensorflow as tf
#import prepare_data
import tarfile
import numpy as np
import pickle
from os.path import join as pjoin
import random

SOURCE_URL = 'https://omnomnom.vision.rwth-aachen.de/data/BiternionNets/'
data_path = "data"
real_filepath = 'TownCentre.pkl.gz'
cg_filepath = 'CGData.pkl.gz'
HOCoffee_zippath = 'HOCoffee-wflip.pkl.gz'
HOCoffee_path = 'HOCoffee.json'


def maybe_download(filename,work_directory):
    if not tf.gfile.Exists(work_directory):
        tf.gfile.MakeDirs(work_directory)

    filepath = os.path.join(work_directory,filename)

    if not tf.gfile.Exists(filepath):
        #filename = "TownCentreHeadImages.tar.bz2"
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        f = tarfile.open(filepath,"r:bz2")
        f.extractall(work_directory)
        print('Successfully downloaded and extracted', filename)

    return filepath


def dump_data(real_use=False,cg_use=False):
    if (not tf.gfile.Exists(os.path.join(data_path,real_filepath))) and real_use:
        prepare_data.prepare_dump(True,False)
    if (not tf.gfile.Exists(os.path.join(data_path,cg_filepath))) and cg_use:
        prepare_data.prepare_dump(False,True)
    if not tf.gfile.Exists(os.path.join(data_path,HOCoffee_zippath)):
        prepare_data.prepare_dump(False,False,True)


class Dataset():
    def __init__(self,filepath):
        X,y,n = pickle.load(gzip.open(filepath, 'rb'))
        if filepath == pjoin(data_path,real_filepath):
            self.train , self.test = self.split_real(X,y,n,trainNum=7800)
            self.train , self.validate =self.split_real(self.train.images*255,self.labels,self.train.n)
        else:
            self.train, self.test = self.split_cg(X,y,n)

    def name_count(self,names,name):
        count = 0
        ind = []
        for i,n in enumerate(names):
            if n == name:
                count += 1
                ind.append(i)

        return count,ind

    def split_real(self,X, y, n, trainNum=7000):
        itr, ite = [], []

        dataNum = X.shape[0]
        # randomIndex = random.shuffle([i for i in range(dataNum)])
        testRate = (dataNum-trainNum)/dataNum

        pid_list = []
        for name in n:
            # Extract the person's ID.
            pid_list.append(int(name.split('_')[1]))

        #IDの重複を除去
        pids = list(set(pid_list))

        #キャラごとのカウントとそのインデックスを計算
        count_list = []
        index_list = []
        for name in pids:
            #名前から数とインデックスの集合を出す(pidsなので重複はしていない)
            c,i = self.name_count(pid_list,name)
            count_list.append(c)
            index_list.append(i)

        maxNum = np.max(np.array(count_list))//2
        #[2,4,6,...]の順に枚数に対応するキャラ数を格納
        chara_num = [count_list.count((i+1)*2) for i in range(maxNum)]
        #枚数ごとにテストで使用するキャラ数
        test_num_list = np.array(chara_num)*testRate//1

        for i,num in enumerate(test_num_list):
            #figNum : 画像の枚数、 num : 必要なキャラ数、 loopcount : 取り出したキャラ数をカウント
            figNum = (i+1)*2
            loopcount = 0
            for j,number in enumerate(count_list):
                # number : あるキャラの枚数（n[]と対応）
                if number == figNum:
                    if loopcount >= num:
                        #　num（必要な数）だけ取り出したら残りは学習データに格納
                        for ind in index_list[j]:
                            itr.append(ind)
                        continue
                    loopcount += 1
                    for ind in index_list[j]:
                        # index_list : キャラごとのインデックスリストのリスト
                        ite.append(ind)

        return data(X[itr], y[itr], [n[i] for i in itr]), data(X[ite], y[ite], [n[i] for i in ite])

    def split_cg(self,X,y,n,split=0.9):
        name = list(set(n))
        trainSize = int(len(name)*split)
        random.shuffle(name)
        trainName = name[:trainSize]
        testName = name[trainSize:]
        trax,tray,tran = [],[],[]
        tesx,tesy,tesn = [],[],[]

        for i,image in enumerate(X):
            if n[i] in trainName:
                trax.append(image)
                tray.append(y[i])
                tran.append(n[i])
            elif n[i] in testName:
                tesx.append(image)
                tesy.append(y[i])
                tesn.append(n[i])
            else:
                print("no such name of cg-object")

        trax = np.array(trax)
        tesx = np.array(tesx)
        tray = np.array(tray)
        tesy = np.array(tesy)

        return data(trax,tray,tran),data(tesx,tesy,tesn)


class data():
    def __init__(self,X,y,n,dtype = tf.float32):
        images = X.astype(numpy.float32)
        self._images = numpy.multiply(images,1.0/255.0)
        self._labels = y
        self.n = n
        self._index_in_epoch = 0
        self._num_examples = X.shape[0]
        self._epochs_completed = 0

    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def suffle(self):
    	# Shuffle the data
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

    def split(self,num):
        self._images = self._images[:num]
        self._labels = self._labels[:num]
        self._num_examples = num


    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
    		# Finished epoch
            self._epochs_completed += 1
    		# Shuffle the data
            self.suffle()

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images[start:end],self._labels[start:end]


class Dataset_cls():
    def __init__(self,filepath):
        Xtr, ytr, ntr, Xte, yte, nte, le = pickle.load(gzip.open(filepath, 'rb'))
        self.train = data(Xtr,ytr,ntr)
        self.test = data(Xte,yte,nte)


def read_data_sets(real_use = True,cg_use = False):
    maybe_download("TownCentreHeadImages.tar.bz2","data")

    dump_data(real_use,cg_use)

    real_data = Dataset(os.path.join(data_path,real_filepath))
    if cg_use and real_use:
        cg_data = Dataset(os.path.join(data_path,cg_filepath))
        return real_data, cg_data

    if real_use:
        return real_data

    if cg_use:
        return cg_data

def read_pickle_data_sets(rNum,cNum):
    realfile = os.path.join("data","realData.pickle")
    cgfile = os.path.join("data","cgData.pickle")

    real_data = pickle.load(open(realfile,"rb"))
    cg_data = pickle.load(open(cgfile,"rb"))

    real_data.train.suffle()
    real_data.train.split(rNum)
    cg_data.train.suffle()
    cg_data.train.split(cNum)

    return real_data,cg_data

def read_hocoffee(fileName):
    maybe_download(HOCoffee_path,data_path)
    dump_data()
    data = Dataset(os.path.join(data_path,HOCoffee_path))

    return data
