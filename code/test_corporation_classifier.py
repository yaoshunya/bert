# -*- coding: utf-8 -*-
#実写でトレーニングし、CGの特徴量を可視化。
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.examples.tutorials.mnist import input_data
# from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import math, os
import pickle
import pdb
import input_data
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import cv2
import sys
from matplotlib.colors import LinearSegmentedColormap
from pylab import rcParams
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd

#===========================
# レイヤーの関数
def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))

def bias_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

# fc layer with ReLU
def fc_relu(inputs, w, b, rate=0.0):
    #pdb.set_trace()
    fc = tf.matmul(inputs, w) + b
    fc = tf.nn.dropout(fc, 1-rate)
    fc = tf.nn.relu(fc)
    return fc


#=========================================================================
def baseNN(x,reuse=False,isTrain=True,rates=[0.0,0.0]):
    node = [1536,768,300,100,1]
    layerNum = len(node)-1
    f_size = 3

    with tf.variable_scope('baseCNN') as scope:
        if reuse:
            scope.reuse_variables()

        W = [weight_variable("convW{}".format(i),[node[i],node[i+1]]) for i in range(layerNum)]
        B = [bias_variable("convB{}".format(i),[node[i+1]]) for i in range(layerNum)]
        fc1 = fc_relu(x,W[0],B[0],rates[1])
        #fc = [fc_relu(fc,W[i+1],B[i+1]) for i in range(layerNum-1)]
        fc2 = fc_relu(fc1,W[1],B[1])
        fc3 = fc_relu(fc2,W[2],B[2])
        fc3 = tf.matmul(fc3,W[3]) + B[3]
    return fc3

#========================================


def next_batch(batch_size,x,y,index_in_epoch,epochs_completed):
    num_examples = x.shape[0]

    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > num_examples:
        epochs_completed += 1
        perm = np.arange(num_examples)
        x = x[perm]
        y = y[perm]
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return x[start:end],y[start:end],x,y,index_in_epoch,epochs_completed

if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    sess = tf.Session(config=config)

    #Epoch数
    nEpo = 500
    # バッチデータ数
    batchSize = 100
    modelPath = 'models'

    #======================================
    # データ読み込み
    X = pickle.load(open("../data/out/data_x.pickle","rb"))
    Y = pickle.load(open("../data/out/data_y.pickle","rb"))

    data_raw = pd.read_csv("../../corporationClassifier/data/corporation_sample.csv")

    title = data_raw['title']
    description = data_raw['description']
    indices = np.arange(len(X))

    train_x,test_x,train_y,test_y,train_index,test_index = train_test_split(X,Y,indices,test_size = 0.1,random_state=0)
    train_y = train_y[np.newaxis].T
    test_y = test_y[np.newaxis].T
    #pdb.set_trace()
    x_train = tf.placeholder(tf.float32,shape=[None,1536])
    x_label = tf.placeholder(tf.float32,shape=[None,1])


    x_test = tf.placeholder(tf.float32,shape=[None,1536])
    x_test_label = tf.placeholder(tf.float32,shape=[None,1])

    #======================================
    #--------------------------------------
    ## build model
    train_pred = baseNN(x_train,rates=[0.2,0.5])

    test_preds = baseNN(x_test,reuse=True,isTrain=False)

    #--------------------------------------
    ## loss function

    train_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=train_pred,labels=x_label))
    test_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=test_preds,labels=x_test_label))

    #--------------------------------------
    ## trainer & vars

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="baseCNN")
    with tf.control_dependencies(extra_update_ops):
        regVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="baseCNN")
        trainerReg = tf.train.AdamOptimizer(1e-3).minimize(train_loss, var_list=regVars)
    #trainerReg = tf.train.AdadeltaOptimizer(1e-3).minimize(lossReg, var_list=regVars)

    #--------------------------------------
    #======================================
    # 保存用
    tra_loss_list = []
    tra_preds_list = []
    tra_label_list = []
    tra_conf_mat_list = []
    tra_auc_list = []
    tra_precision_list = []
    tra_recall_list = []

    tes_loss_list = []
    tes_preds_list = []
    tes_label_list = []
    tes_conf_mat_list = []
    tes_auc_list = []
    tes_precision_list = []
    tes_recall_list = []

    ite_list = []
    #======================================
    # 初期化
    sess.run(tf.global_variables_initializer())
    ite = 0
    isStop = False
    epochs_completed = 0
    index_in_epoch = 0
    #======================================

    saver = tf.train.Saver()


    while not isStop:
        ite = ite + 1
        ite_list.append(ite)
        #-----------------
        ## バッチの取得
        batch_x,batch_label,train_x,train_y,index_in_epoch,epochs_completed = next_batch(batchSize,train_x,train_y,index_in_epoch,epochs_completed)
        batch_label = np.reshape(batch_label,[batchSize,1])
        #-----------------

        # training
        _,lossReg_value,pred_value = sess.run([trainerReg,train_loss,train_pred],feed_dict={x_train:batch_x, x_label:batch_label})

        #
        pred_value = (pred_value > 0.5) * 1

        #
        tra_conf_mat = confusion_matrix(batch_label, pred_value)
        tra_auc = roc_auc_score(batch_label, pred_value)
        tra_precision = precision_score(batch_label,pred_value)
        tra_recall = recall_score(batch_label,pred_value)

        # 保存
        tra_loss_list.append(lossReg_value)
        tra_preds_list.append(pred_value)
        tra_label_list.append(batch_label)
        tra_conf_mat_list.append(tra_conf_mat)
        tra_auc_list.append(tra_auc)
        tra_precision_list.append(tra_precision)
        tra_recall_list.append(tra_recall)
        # test
        test_pred_value, test_lossReg_value = sess.run([test_preds,test_loss],feed_dict={x_test:test_x, x_test_label:test_y})

        #
        test_pred_value = (test_pred_value > 0.5) * 1

        #
        tes_conf_mat = confusion_matrix(test_y, test_pred_value)
        tes_auc = roc_auc_score(test_y, test_pred_value)
        tes_precision = precision_score(test_y,test_pred_value)
        tes_recall = recall_score(test_y,test_pred_value)
        #pdb.set_trace()

        #
        if ite%500==0:
           saver.save(sess,f"{modelPath}/model_{ite}.ckpt")
           print("ite{0}:trainLoss:{1},testLoss:{2}".format(ite,lossReg_value,test_lossReg_value))
           #print("       confusion matrix : train {0}, test {1}".format(tra_conf_mat,tes_conf_mat))
           print("       auc              : train {0}, test {1}".format(tra_auc,tes_auc))
           print("       precision        : train {0}, test {1}".format(tra_precision,tes_precision))
           print("       recall           : train {0}, test {1}".format(tra_recall,tes_recall))

        # 保存
        tes_loss_list.append(test_lossReg_value)
        tes_preds_list.append(test_pred_value)
        tes_conf_mat_list.append(tes_conf_mat)
        tes_auc_list.append(tes_auc)
        tes_precision_list.append(tes_precision)
        tes_recall_list.append(tes_recall)

        if epochs_completed==nEpo:
            isStop = True

    train_len = len(train_x)
    #pdb.set_trace()
    print("train confusion matrix:\n{0}".format(tra_conf_mat))
    print("test confusion matrix :\n{0}".format(tes_conf_mat))
    df = pd.DataFrame({
            'title':title[test_index],
            'description':description[test_index],
            'true class':test_y.T[0],
            'predict class':test_pred_value.T[0]
            })
    df = df[['title','description','true class','predict class']]
    #pdb.set_trace()
    df.to_csv('../data/out/test_result.csv')


    with open("../data/out/log/test_corpolation_classifier_log.pickle","wb") as f:
        pickle.dump(tra_loss_list,f)
        #pickle.dump(tra_preds_list,f)
        pickle.dump(tra_conf_mat_list,f)
        pickle.dump(tra_auc_list,f)
        pickle.dump(tra_precision_list,f)
        pickle.dump(tra_recall_list,f)

        pickle.dump(tes_loss_list,f)
        #pickle.dump(tes_preds_list,f)
        pickle.dump(tes_conf_mat_list,f)
        pickle.dump(tes_auc_list,f)
        pickle.dump(tes_precision_list,f)
        pickle.dump(tes_recall_list,f)
