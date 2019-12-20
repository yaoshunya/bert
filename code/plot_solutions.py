import os
import numpy as np
import pickle
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

if __name__ == "__main__":

    data = []
    dataN = 5
    dataType = 2
    data_name = ["train loss","train confusion matrix","train auc","train precision","train recall","test loss","test confusion matrix","test auc","test precision","test recall"]

    with open("../data/out/log/test_corpolation_classifier_log.pickle","rb") as f:
        for i in range(dataN*dataType):
            data.append(pickle.load(f))

    ite = len(data[0])

    for i in range(len(data_name)):
        plt.close()
        #pdb.set_trace()
        if data_name[i] == "train confusion matrix" or data_name[i] == "test confusion matrix":
            #pdb.set_trace()
            continue
        plt.plot(range(ite),data[i])
        if data_name[i] == "train loss" or data_name[i] == "test loss":
            plt.ylim([0,2])
        else:
            plt.ylim([0.5,1.1])
        plt.xlabel("iteration")
        plt.ylabel(data_name[i])
        plt.savefig("../data/out/{0}.png".format(data_name[i]))
