#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2017-09-06 16:48
# * Last modified : 2017-10-24 11:39
# * Filename      : ilf-logClassification.py
# * Description   :
'''

The difference of this version and multi version is reading data. This version has added preprocess before save label in list.

This version can save labels, results and logs
If change to tfilf, --add_ilf
'''
# **********************************************************

from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import matplotlib
matplotlib.use('Agg')
import logging
import numpy as np
import math
from optparse import OptionParser
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
#from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from puLearning.puAdapter import PUAdapter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

# import mail_mwb

import sys

def sendEmail(message='',title=''):
        import time
        mail_message=sys.argv[0]+' has been finished in '+str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        mail_message+=' \n total time:'+str(mail_time-time.time()/60)+' mins'
        if message!='':
            mail_message=message+' '+mail_message
        if title=='':
            title=sys.argv[0]+' has finished'
        mail_mwb.sentEmail(mail_message,title)


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--add_length",
              action="store_true", dest="add_length_vector",
              help="Add length vector")
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--add_ilf",
	      action="store_true", dest="add_ilf",
	      help="add_ilf")

op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")

def testPU(X_train,y_train,X_test,y_test,total_pred_y_pu,total_pred_y_re):
    np.random.seed(42)

    pu_pred=[]
    # print "Loading dataset"
    # print

    # #Shuffle dataset
    # print "Shuffling dataset"
    # print
    # #如果传给permutation一个矩阵，它会返回一个洗牌后的矩阵副本
    # #如果传入一个整数，它会返回一个洗牌后的arange。

    # permut = np.random.permutation(len(y))
    # X = X[permut]
    # y = y[permut]

    #make the labels -1.,+1.


    # y_train[np.where(y_train == 0)[0]] = 0
    # y_train[np.where(y_train == 1)[0]] = +1.
    permut  = np.random.permutation(len(y_train))
    X_train = X_train[permut]
    y_train = y_train[permut]
    # y_test[np.where(y_test == 0)[0]] = 0
    # y_test[np.where(y_test == 1)[0]] = +1.


    # print "Loaded ", len(y), " examples"
    # print len(np.where(y == -1.)[0])," are bening(unlabled)"
    # print len(np.where(y == +1.)[0])," are malignant"
    # print

    # #Split test/train
    # print "Splitting dataset in test/train sets"
    # print
    # split = 2*len(y)/3
    # X_train = X[:split]
    # y_train = y[:split]
    # X_test = X[split:]
    # y_test = y[split:]

    print("Training set contains ", len(y_train), " examples")
    print (len(np.where(y_train == -1.)[0])," are bening(unlabled)")
    print (len(np.where(y_train == 1.)[0])," are malignant")
    print()
    isEmpty=False
    pu_f1_scores = []
    reg_f1_scores = []
    step=len(np.where(y_train == 1.)[0])/(pu_iter_time*multiple_for_pu_iter_time+1)
    n_sacrifice_iter=[]
    for i in range(pu_iter_time):
        n_sacrifice_iter.append(i*step)
    # n_sacrifice_iter = range(0, len(np.where(y_train == +1.)[0])-21, 20)
    print("n_sacrifice_iter",n_sacrifice_iter)
    if total_pred_y_pu==[]:
        isEmpty=True
    for i,n_sacrifice in enumerate(n_sacrifice_iter):
        print('=' * 80)
        #send some positives to the negative class! :)
        print("PU transformation in progress.")
        print("Making ", n_sacrifice, " malignant examples bening.")
        print()
        y_train_pu = np.copy(y_train)
        pos = np.where(y_train == 1.)[0]
        np.random.shuffle(pos)
        sacrifice = pos[:n_sacrifice]
        y_train_pu[sacrifice] = -1.

        print( "PU transformation applied. We now have:")
        print( len(np.where(y_train_pu == -1.)[0])," are bening")
        print( len(np.where(y_train_pu == 1.)[0])," are malignant")
        print('-' * 80)

        #Get f1 score with pu_learning
        print( "PU learning in progress...")
        estimator = RandomForestClassifier(n_estimators=10, # #_of_submodel
                                           criterion='entropy', #gini or entropy(default=”gini”)是计算属性的gini(基尼不纯度)还是entropy(信息增益)，来选择最合适的节点。
                                           bootstrap=True, #bootstrap=True：是否有放回的采样。
                                           n_jobs=1) #并行job个数   1=不并行；n：n个并行；-1：CPU有多少core，就启动多少job
        pu_estimator = PUAdapter(estimator)
        pu_estimator.fit(X_train,y_train_pu)
        y_pred = pu_estimator.predict(X_test)
        if isEmpty:
            total_pred_y_pu.append(list(y_pred))
        else:
            total_pred_y_pu[i].extend(list(y_pred))
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
        pu_f1_scores.append(f1_score[1])
        print ("F1 score: ", f1_score[1])
        print ("Precision: ", precision[1])
        print ("Recall: ", recall[1])
        print()

        #Get f1 score without pu_learning
        print( "Regular learning in progress...")
        estimator = RandomForestClassifier(n_estimators=10,
                                           bootstrap=True,
                                           n_jobs=1)
        estimator.fit(X_train,y_train_pu)
        y_pred = estimator.predict(X_test)
        if isEmpty:
            total_pred_y_re.append(list(y_pred))
        else:
            total_pred_y_re[i].extend(list(y_pred))
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
        reg_f1_scores.append(f1_score[1])
        print( "F1 score: ", f1_score[1])
        print( "Precision: ", precision[1])
        print ("Recall: ", recall[1])

        #testing
        # print( "old learning in progress...")
        # results=[]
        # results.append(benchmark(LinearSVC(penalty="l2", dual=False,tol=1e-3),X_train,y_train_pu,X_test,y_test))
        print('=' * 80)
        print('\n')

    # plt.title("Random forest with/without PU learning")
    # plt.plot(n_sacrifice_iter, pu_f1_scores, label='PU Adapted Random Forest')
    # plt.plot(n_sacrifice_iter, reg_f1_scores, label='Random Forest')
    # plt.xlabel('Number of positive examples hidden in the unlabled set')
    # plt.ylabel('F1 Score')
    # plt.legend()
    # plt.show()

    # return total_pred_y_pu

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."
def ngramDivide(n,inputData):
    '''
        divide log into n-gram

        return: vocabulary (word list)
    '''
#    print("cutting gram")
    vocabulary=set()
    for line in inputData:
        l=line.strip().split()
        cur_len=len(l)
        cur_index=0
	if cur_len==0:
            continue
        #if cur_len==1:
        #    cur_gram=l[0]
	#    vocabulary.add(cur_gram)
	if cur_len<n:
	    cur_gram=' '.join(l)
            vocabulary.add(cur_gram)
            #print('!!!!!!!cur_len<n!!!!!!!')
	    continue
        loop_num=cur_len-n+1
        for i in range(loop_num):
            cur_gram=' '.join(l[i:i+n])
	    vocabulary.add(cur_gram)
    #print(vocabulary)
    #print(len(vocabulary))
#    print("cutting gram end")
    return list(vocabulary)

def convertLogToVector(n,inputData,vocabulary,y):
    result=[]
    x_result=[]
    y_result=[]

#    print("convertLogToVector start")
    for index_data,line in enumerate(inputData):
	l=line.strip().split()
	temp=[]
        cur_len=len(l)
        cur_index=0
        if cur_len==0:
            continue
        #if cur_len==1:
        #    cur_gram=l[0]
	#    temp.append(vocabulary.index(cur_gram))
        if cur_len<n:
            cur_gram=' '.join(l)
            if cur_gram not in vocabulary:
                continue
            else:
                temp.append(vocabulary.index(cur_gram))
		x_result.append(line)
		y_result.append(y[index_data])
            #print('!!!!!!!cur_len<n!!!!!!!')
            result.append(temp)
            continue
        loop_num=cur_len-n+1
        for i in range(loop_num):
            cur_gram=' '.join(l[i:i+n])
            if cur_gram not in vocabulary:
	        continue
	    else:
	        temp.append(vocabulary.index(cur_gram))
	result.append(temp)
	x_result.append(line)
	y_result.append(y[index_data])
#    print("convertLogToVector end")
    return np.array(result),np.array(y_result),x_result


def calculateTfidfForTrain(inputVector,vocabulary):
    '''
        In this version, tf is not normalized. We use frequence value as tf value.

	    RETRUN: tfidf,tfidf_mean,tfidf_std,idf_dict
    '''
    tfidf=[]
    idf_dict={}
    ilf_dict={}
    gram_index_ilf_dict={}
    # print(len(inputVector))

    t0=time()

    total=len(inputVector)
    divideLap=10
    ded=total/divideLap
    t1=time()
    total_log_num=len(inputVector)
    gram_index_dict={}#gram_index_dict for idf
    max_longth=0
    for index,l in enumerate(inputVector):
        if len(line)>max_longth:
            max_longth=len(l)
	if index%ded==0:
	    #print("  "+str(index/ded)+'/10 '+str(time()-t1))
	    t1=time()
	for location,gram in enumerate(l) :
            if gram not in gram_index_dict:
                gram_index_dict[gram]=set()
	    gram_index_dict[gram].add(index)
	    if gram not in gram_index_ilf_dict:
		gram_index_ilf_dict[gram]=set()
	    gram_index_ilf_dict[gram].add(location)

	    #if index not in gram_index_dict[gram]:
	    #gram_index_dict[gram].append(index)
    #print(" t1:"+str(time()-t0))
    t0=time()
    #calculating idf
    total=len(gram_index_dict)
    divideLap=10
    ded=total/divideLap
    t1=time()
    for index,gram in enumerate(gram_index_dict):
	if index%ded==0:
            #print("  "+str(index/ded)+'/10 '+str(time()-t1))
            t1=time()
        cur_idf=math.log(float(total_log_num)/(float( len(gram_index_dict[gram]) +0.01 )))

	cur_ilf=math.log(float(max_longth)/(float( len(gram_index_ilf_dict[gram]) +0.01 )))
	idf_dict[gram]=cur_idf
	ilf_dict[gram]=cur_ilf
        # print(len(gram_index_dict[gram]),cur_idf)
#    print("t1:"+str(time()-t0))
    #print(" t2:"+str(time()-t0))
    t0=()
#    #calculating tfidf
#    for index,l in enumerate(inputVector):
#        cur_tf=[]
#        for j,gram in enumerate(vocabulary):
#            num=0
#            for w in l:
#                if w==j:
#                    num+=1
#                    # print ("ssssssssss")
#            cur_tf.append(float(num)*idf_dict[j])
#
#	tfidf.append(np.array(cur_tf))
#    tfidf=np.array(tfidf)
    t0=time()
    total=len(inputVector)
    divideLap=10
    ded=total/divideLap
    t1=time()
    for index,l in enumerate(inputVector):
	if index%ded==0:
	    # print("  "+str(index/ded)+'/10 '+str(time()-t1))
	    t1=time()
	cur_tfidf=np.zeros(len(vocabulary))
        for gram_index in l:
	    if opts.add_ilf:
                cur_tfidf[gram_index]=float(l.count(gram_index))*ilf_dict[gram_index]
                #cur_tfidf[gram_index]=float(l.count(gram_index))*idf_dict[gram_index]*ilf_dict[gram_index]
            else:
                cur_tfidf[gram_index]=float(l.count(gram_index))*idf_dict[gram_index]
	    #cur_tfidf[gram_index]=float(l.count(gram_index))*idf_dict[gram_index]*ilf_dict[gram_index]
        tfidf.append(cur_tfidf)
#    print("t2:"+str(time()-t0))
    tfidf=np.array(tfidf)
    #print(" t3:"+str(time()-t0))
#    tfidf_mean=np.mean(tfidf,axis=0)
#    tfidf_std=np.std(tfidf,axis=0)

    #calculating normalized tfidf
    # tfidf=np.true_divide((tfidf-tfidf_mean),tfidf_std)



    return tfidf,idf_dict,ilf_dict

def calculateTfidfForTest(idf_dict,ilf_dict,inputVector,vocabulary):
    tfidf=[]
    #calculating tfidf
    t0=time()
    for index,l in enumerate(inputVector):
        cur_tfidf=np.zeros(len(vocabulary))
        for gram_index in l:
            if opts.add_ilf:
                cur_tfidf[gram_index]=float(l.count(gram_index))*ilf_dict[gram_index]
		#cur_tfidf[gram_index]=float(l.count(gram_index))*idf_dict[gram_index]*ilf_dict[gram_index]
	    else:
		cur_tfidf[gram_index]=float(l.count(gram_index))*idf_dict[gram_index]
        tfidf.append(cur_tfidf)
#    print("t2:"+str(time()-t0))
    tfidf=np.array(tfidf)

    #calculating normalized tfidf
    #tfidf=np.true_divide((tfidf-tfidf_mean),tfidf_std)
    return tfidf

def addLengthInFeature(X,X_train_bag_vector):
    '''
        X_train,max_length=addLengthForTrain(X_train,X_train_bag_vector)
        Return: X_train_with ,max_length
    '''
    len_list=[]
    for line in X_train_bag_vector:
        len_list.append(len(line))
    len_final=len(len_list)
    len_list=np.array(len_list).reshape(len_final,1)
    # print(X_train.shape)
    # print(len_list.shape)
    X=np.hstack((X,len_list))
    # print(X_train.shape)
    return X



# #############################################################################
# Benchmark classifiers
def benchmark(clf,X_train,y_train,X_test,y_test):
    print('_' * 80)
    print("LinearSVC Training: ")
    # print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                print(i,label)
                length=min(len(clf.coef_[i]),10)
                top10 = np.argsort(clf.coef_[i])[-length:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
		print()

    # if opts.print_report:
    #     print("classification report:")
    #     print(metrics.classification_report(y_test, pred,
    #                                         target_names=target_names))

    # if opts.print_cm:
    #     print("confusion matrix:")
    #     print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    pred=list(pred)
    # print(pred)


    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, pred)
    print( "F1 score: ", f1_score)
    print( "Precision: ", precision)
    print ("Recall: ", recall)


    return clf_descr, score, train_time, test_time, pred
    #results = [[x[i] for x in results] for i in range(5)]



if __name__ == '__main__':

    mail_time=time()
    input_path='./data/data_for_binary_detection.dat'
    # fig_path='./RSA_sklearn.png'
    k_of_kflod=3
    n_for_gram=1
    unlabel_label="Unlabel"
    pu_iter_time=5 # iters of pu test
    multiple_for_pu_iter_time=2 # step=len(np.where(y_train == 1.)[0])/(pu_iter_time*multiple_for_pu_iter_time+1)
    lable_result_log_filename='pulearning_label_result_log.dat'
    pu_fscore_file='pu_fscore_file'
    argv = [] if is_interactive() else sys.argv[1:]
    (opts, args) = op.parse_args(argv)
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

    print(__doc__)
    op.print_help()
    print()


    # #############################################################################

    t_start=time()
    X_data=[]
    label_dict={}
    y_data=[]
    target_names=[]
    with open(input_path) as IN:
        for line in IN:
            l=line.strip().split()
            label=l[0]
            if label != unlabel_label:
                label='Anomaly'

            if label not in label_dict:
                if label == unlabel_label:
                    label_dict[label]=-1.
                else:
                    label_dict[label]=1.
                target_names.append(label)
            X_data.append(' '.join(l[2:]))
            y_data.append(label_dict[label])
    X_data=np.array(X_data)
    y_data=np.array(y_data)
    # print(set(y_data))
    y_test=[]
    y_train=[]
    X_test=[]
    X_train=[]

    #KFold
    skf=StratifiedKFold(n_splits=k_of_kflod)
    skf.get_n_splits(X_data,y_data)
    cur_num=1

    clf_names_list=[]
    training_time_list=[]
    test_time_list=[]
    pred_list=[]
    y_list=[]
    x_save_list=[]


    total_pred_y_pu=[]
    total_pred_y_re=[]
    for train_index,test_index in skf.split(X_data,y_data):
        print('=' * 80)
        print('\ncur_iteration:%d/%d' % (cur_num,k_of_kflod))
        cur_num+=1
        X_train,X_test=X_data[train_index],X_data[test_index]
        y_train,y_test=y_data[train_index],y_data[test_index]
        print(" train data size:"+str(X_train.shape[0]))
        print(" test  data size:"+str(X_test.shape[0] ))

        #new code#######################
        #vocabulary is a list including words
        t0=time()
        print(" ngramDivide start")
        vocabulary=ngramDivide(n_for_gram,X_train)
        print("  ngramDivide end, time="+str(time()-t0)+"s")

        t0=time()
        print(" convertLogToVector for train start")
        X_train_bag_vector,y_train,X_train_save=convertLogToVector(n_for_gram,X_train,vocabulary,y_train)
        print("  convertLogToVector for train end, time="+str(time()-t0)+"s")
        print(" X_train_bag_vector.shape:"+str(X_train_bag_vector.shape))
        t0=time()
        print(" convertLogToVector for test start")

        #print(len(X_test),len(y_test))
        X_test_bag_vector,y_test,X_test_save=convertLogToVector(n_for_gram,X_test,vocabulary,y_test)
        print("  convertLogToVector for test end, time="+str(time()-t0)+"s")

        t0=time()
        print(" calculateTfidfForTrain start")
        X_train,idf_dict,ilf_dict=calculateTfidfForTrain(X_train_bag_vector,vocabulary)
        print("  calculateTfidfForTrain end, time="+str(time()-t0)+"s")
        print(" X_train.shape:"+str(X_train.shape))

        t0=time()
        print(" calculateTfidfForTest start")
        X_test=calculateTfidfForTest(idf_dict,ilf_dict,X_test_bag_vector,vocabulary)
        print("  calculateTfidfForTest end, time="+str(time()-t0)+"s")
        #print(X_train.shape)
        #print(X_test.shape)
        y_test=np.array(y_test)
        y_train=np.array(y_train)
        y_list.append(y_test)
        x_save_list.append(X_test_save)
        #add length to feature vector
        if opts.add_length_vector:
            print(" Adding length as feature")
            X_train=addLengthInFeature(X_train,X_train_bag_vector)
            X_test=addLengthInFeature(X_test,X_test_bag_vector)
            print("  X_train.shape after add lengeth feature:"+str(X_train.shape))

        feature_names=vocabulary

        if feature_names:
            feature_names = np.asarray(feature_names)

        results = []

        print('=' * 80)
        testPU(X_train,y_train,X_test,y_test,total_pred_y_pu,total_pred_y_re)
        print('=' * 80)
        results.append(benchmark(LinearSVC(penalty="l2", dual=False,tol=1e-3),X_train,y_train,X_test,y_test))


    total_y=[]
    for k in y_list:
        total_y.extend(k)

    #total_pred_y_pu
    total_x_save=[]
    for k in x_save_list:
        total_x_save.extend(k)

    if opts.add_length_vector:
        lable_result_log_filename='./'+str(k_of_kflod)+'_'+str(n_for_gram)+'_add_'+lable_result_log_filename
    else:
        lable_result_log_filename='./'+str(k_of_kflod)+'_'+str(n_for_gram)+'_'+lable_result_log_filename

    f_yyx = open(lable_result_log_filename,'w')


    file_score = open(pu_fscore_file+'_'+str(multiple_for_pu_iter_time)+'_'+str(pu_iter_time)+'_'+str(k_of_kflod)+'_total_'+str(len(y_data))+'.dat','w')
    for i,n in enumerate(total_pred_y_pu):
            #precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
            print('=' * 80)
            print(str(i)+'/'+str(pu_iter_time))
            # print("puLearning classification report:" )

            # print(metrics.classification_report(total_y, total_pred_y_pu[i],
            #                                     target_names=target_names))
            # print("confusion matrix:" )
            # print(metrics.confusion_matrix(total_y, total_pred_y_pu[i]))
            precision1, recall1, f1_score1, _ = precision_recall_fscore_support(total_y, total_pred_y_pu[i])
            print("puLearning precision:"+str(round(precision1[1],4))+ " recall:"+str(round(recall1[1],4))+" f1_score:"+str(round(f1_score1[1],4)))
            print('-' * 80)
            # print(str(i)+'/'+str(pu_iter_time))
            # print("regular classification report:" )
            # print(metrics.classification_report(total_y, total_pred_y_re[i],
            #                                     target_names=target_names))

            # print("confusion matrix:" )
            # print(metrics.confusion_matrix(total_y, total_pred_y_re[i]))
            precision2, recall2, f1_score2, _ = precision_recall_fscore_support(total_y, total_pred_y_re[i])
            print("regular precision:"+str(round(precision2[1],4))+ " recall:"+str(round(recall2[1],4))+" f1_score:"+str(round(f1_score2[1],4)))
            file_score.writelines(str(f1_score1[1])+' '+str(f1_score2[1])+' '+str(int(i*len(y_data)/(multiple_for_pu_iter_time*pu_iter_time)))+'\n')

            print('=' * 80)


    score=[]
    #print accuracy
    print('=' * 80)
    for i,n in enumerate(clf_names_list):
        # print('=' * 80)
        cur_score = metrics.accuracy_score(total_y, pred_list[i])
        score.append(cur_score)
        print("%s accuracy:   %0.3f" % (n,cur_score))
    print('=' * 80)

    for i,k in enumerate(total_y):
        pred = ""
        for j in range(len(pred_list)):
    	   pred += ' ' + str(pred_list[j][i])
        f_yyx.writelines(target_names[int(k)] + " " +str(k) + ' ' + pred + ' ' + total_x_save[i]+'\n')

    print('total time: '+str(int((time()-t_start)))+'s')
