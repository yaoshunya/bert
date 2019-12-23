import pandas as pd
import pdb
import numpy as np
import tensorflow as tf
import MeCab
from gensim.models import word2vec
import pickle
from bert_juman import BertWithJumanModel

if __name__ ==  '__main__':

    df = pd.read_csv("../../corporationClassifier/data/data_all.csv") #load csv file

    bert = BertWithJumanModel("../model/Japanese_L-12_H-768_A-12_E-30_BPE") #モデルの読み込み

    title = df['title'] #タイトルの全データ
    description = df['description'] #説明の全データ
    y = df['class'] #0 or 1
    data_len = title.shape[0]

    out_title = [] #タイトルの全データの特徴量が入る
    out_description = [] #内容の全データの特徴量が入る

    #------------------------------------
    #データ数分forループ
    for i in range(data_len):

        parts_title = [] #タイトルごとの特徴量が入る
        parts_description = [] #内容ごとの特徴量が入る

        #-------------------------
        #タイトルに関して特徴量を抽出
        #-------------------------
        parts_title = bert.get_sentence_embedding(title[i])
        #-------------------------
        #説明に関して特徴量を抽出
        #-------------------------
        parts_description = bert.get_sentence_embedding(description[i])
        #-------------------------

        out_title.append(parts_title) #タイトルごとに配列に追加
        out_description.append(parts_description) #説明ごとに配列に追加

        print(i)
    #-------------------------


    for i in range(len(out_title)):

        feature_stack = np.concatenate([np.array(out_title[i]),np.array(out_description[i])])

        if i == 0:
            X = feature_stack[np.newaxis]
        else:
            X = np.append(X,feature_stack[np.newaxis],axis=0)

    Y = np.array(y) #labelをnumpyに変換

    #-------------------------

    #-------------------------
    #------データをpickleで保存
    with open('../data/out/data_x.pickle','wb') as f:
        pickle.dump(X,f)
    with open('../data/out/data_y.pickle','wb') as f:
        pickle.dump(Y,f)
    #-------------------------

    #-------------------------

    #------------------------------------
