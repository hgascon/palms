#!/usr/bin/python
# PALMS Data Partition and Automatic LibSVM Model Selection
# Copyright (c) 2011 Hugo Gascon <hgascon@gmail.com>

import sys
import os
import random
import gzip
import operator
import numpy as np
import matplotlib.pyplot as plt
from svmutil import *
from mleval import *
from optparse import OptionParser
from eval import *
from roc import *

def get_samples(dic_labels, dic_features, label):
   """ Return all data and labels from one class """
   
   samples = [v for k, v in dic_features.items() if dic_labels[k] == label]
   return samples

def make_binary(labels, l):
    """ Change a multiclass vector into a binary vector according label l"""

    bin_labels = []
    for value in labels:
        if value == l:
            bin_labels.append(np.float64(1))
        else:
            bin_labels.append(np.float64(-1))
        
    return bin_labels

def load_svm_file(svm_file):
    """ Load labels, features and index dictionaries for both from a LibSVM file """

    #Load data from file
    labels = list(set(np.loadtxt(svm_file, usecols=[0], dtype = float)))
    y, x = svm_read_problem(svm_file)
    dic_labels = dict(zip(range(0,len(y)),y))
    dic_features = dict(zip(range(0,len(x)),x))

    return y, x, dic_labels, dic_features, labels

def split_class(dic_labels, dic_features,c,l):
    """ Split data from class c in training, validate and test according known class l """

    if c == l:
        class_select = float(1)
    else:
        class_select = float(-1)
    
    samples = get_samples(dic_labels, dic_features, c)
    K = len(samples)/3
    r = random.randint(0,len(samples))
    s2 = samples + samples + samples

    x_train = s2[r:r+K]
    y_train = list(np.ones(len(x_train)) * class_select)

    x_validate = s2[r+K:r+2*K]
    y_validate = list(np.ones(len(x_validate)) * class_select)
    y_validate_labels = [c] * len(x_validate)
    
    x_test = s2[r+2*K:r+3*K]
    y_test = list(np.ones(len(x_test)) * class_select)
    y_test_labels = [c] * len(x_test)

    data = [x_train, y_train,
            x_validate, y_validate, y_validate_labels,
            x_test, y_test, y_test_labels]

    return data

def split_data_basic(y,x,dic_labels,dic_features,l,labels):
    """ Split data from class l in training, validate and test sets in basic mode. """

    #select some classes be used as unknown classes (label -1) 
    unknown_labels = list(labels)
    unknown_labels.remove(l)

    x_train = []
    y_train = []
    x_validate = []
    y_validate = []
    y_validate_labels = []
    x_test = [] 
    y_test = []
    y_test_labels = []

    #Build known class data
    data = split_class(dic_labels, dic_features,l,l)
    x_train.extend(data[0])
    y_train.extend(data[1])
    x_validate.extend(data[2])
    y_validate.extend(data[3])
    y_validate_labels.extend(data[4])
    x_test.extend(data[5])
    y_test.extend(data[6])
    y_test_labels.extend(data[7])

    #Build unknown class data
    for u_l in unknown_labels:
        data = split_class(dic_labels, dic_features,u_l,l)
        x_train.extend(data[0])
        y_train.extend(data[1])
        x_validate.extend(data[2])
        y_validate.extend(data[3])
        y_validate_labels.extend(data[4])
        x_test.extend(data[5])
        y_test.extend(data[6])
        y_test_labels.extend(data[7])

    data = [x_train, y_train,
            x_validate, y_validate, y_validate_labels,
            x_test, y_test, y_test_labels]

    return data

def split_data_advanced(y,x,dic_labels,dic_features,l,labels):
    """ Split data from class l in training, validate and test sets in advanced mode. """

    #select some classes be used as unknown classes (label -1) 
    unknown_labels = list(labels)
    unknown_labels.remove(l)
    unknown_training_labels = random.sample(unknown_labels,len(unknown_labels)/3)
    unknown_val_test_labels = list(set(labels) - set(unknown_training_labels) - set([l]))
    unknown_validate_labels = random.sample(unknown_val_test_labels, len(unknown_val_test_labels)/2)
    unknown_test_labels = list(set(unknown_val_test_labels) - set(unknown_validate_labels))

    x_train = []
    y_train = []
    x_validate = []
    y_validate = []
    y_validate_labels = []
    x_test = [] 
    y_test = []
    y_test_labels = []

    #Build known class data
    data = split_class(dic_labels, dic_features,l,l)
    x_train.extend(data[0])
    y_train.extend(data[1])
    x_validate.extend(data[2])
    y_validate.extend(data[3])
    y_validate_labels.extend(data[4])
    x_test.extend(data[5])
    y_test.extend(data[6])
    y_test_labels.extend(data[7])

    #Build unknown class train data
    for u_l in unknown_training_labels:
        samples = get_samples(dic_labels, dic_features, u_l)
        x_train.extend(samples)
        y_train.extend(list(np.ones(len(samples)) * -1))

    #Build unknown class validate data
    for u_l in unknown_validate_labels:
        samples = get_samples(dic_labels, dic_features, u_l)
        x_validate.extend(samples)
        y_validate.extend(list(np.ones(len(samples)) * -1))
        y_validate_labels.extend([u_l] * len(samples))

    #Build unknown class test data
    for u_l in unknown_test_labels:
        samples = get_samples(dic_labels, dic_features, u_l)
        x_test.extend(samples)
        y_test.extend(list(np.ones(len(samples)) * -1))
        y_test_labels.extend([u_l] * len(samples))

    data = [x_train, y_train,
            x_validate, y_validate, y_validate_labels,
            x_test, y_test, y_test_labels]

    return data

def model_learning(svm_file, out_file, mode):
    """ Automatic partition, training, validation and testing of multiclass libSVM data """
    
    y, x, dic_labels, dic_features, labels = load_svm_file(svm_file)

    #Set of params for model testing
    c_values = np.logspace(-3,3,7)
    nu_values = np.arange(0.1,1,0.1)
    g_values = np.logspace(-3,3,7)

    class_results = {}
    fs = gzip.open(out_file+".scores","wb")

    for l in labels:

        dic_test = {}
        #y_bin = make_binary(y,l)

        for i in range(10):
            if mode == 0:
                #data = split_data_basic(y_bin, y, x)
                data = split_data_basic(y,x,dic_labels,dic_features,l,labels)
            if mode == 1:
                data = split_data_advanced(y,x,dic_labels,dic_features,l,labels)

            [x_train, y_train, x_validate, 
                    y_validate, y_validate_labels,
                    x_test, y_test, y_test_labels] = data

            dic_validation = {}
            for c in c_values:
                for g in g_values:

				    #train in K keystrokes
                    model = svm_train(y_train, x_train,
                                      "-q -s 0 -t 2 -g "+str(g)+" -c "+str(c))

				    #predict on validation with c,g
                    p_label, p_acc, p_val = svm_predict(y_validate, x_validate, model)

                    fpc, tpc, auc = calculate_auc(np.array(p_val, dtype=float).flatten(),
                                                  np.array(y_validate, dtype=int))
                    dic_validation[(c,g)] = auc 

		    #get best cb,gb
            auc_list = [ v for v in dic_validation.values() if not np.isnan(v)]
            if len(auc_list) != 0:
                auc_max = max(auc_list)
                for j in dic_validation.keys():
                    if dic_validation[j] == auc_max:
                        c_best = j[0]
                        g_best = j[1]
                
                #predict on testing data with best c and g
                test_model = svm_train(y_train, x_train,
                                       "-q -s 0 -t 2 -g "+str(g_best)+" -c "+str(c_best))
                #predict on test with c_best,g_best
                p_label, p_acc, p_val = svm_predict(y_test, x_test, test_model)

                fpc, tpc, auc = calculate_auc(np.array(p_val, dtype=float).flatten(),
                                              np.array(y_test, dtype=int))
                    
                #Original class labels, binary labels and decision values are saved for aggregation
                fs.write("class %s iter %s c %s g %s auc %s test\n" % (l,i,c_best,g_best,auc))
                fs.write(str(y_test_labels)+"\n")
                fs.write(str(y_test)+"\n")
                fs.write(str(p_val)+"\n")

                #save l,r,AUC,accuracy, cb y gb
                dic_test[i] = [c_best,g_best,auc]
                print "Class %s iter %s c %s g %s auc %s" % (l,i,c_best,g_best,auc)
        #save iteration results for this class
        class_results[l] = dic_test

    fs.close()
    #All restults are saved as a dictionary
    fd = open(out_file,'w')
    fd.write(str(class_results))
    fd.close()
    
    return class_results

def model_selection(dic_results):
    """ Select best libSVM model from several iterations and average test AUC """

    print "Label | C | G | Average AUC | Persistence\n"

    #loop across different classes and extract params
    for label in dic_results.keys():
        dic_params={}
        for r in dic_results[label].values():
            if dic_params.has_key((r[0],r[1])):
               if not np.isnan(r[2]):
                   dic_params[(r[0],r[1])][0] += float(r[2])
               dic_params[(r[0],r[1])][1] += 1
            else:
               if not np.isnan(r[2]):
                   dic_params[(r[0],r[1])] = [float(r[2]),1]
               else:
                   dic_params[(r[0],r[1])] = [0,1]
        
        #get max number of parameters repetition
        m = max([v[1] for v in dic_params.values()])
        
        #print results 
        for item in dic_params.items(): 
            if item[1][1] == m :
                model = str(item[0][0])+" | "+str(item[0][1])+" | "+str(item[1][0]/m)+" | "+str(m)+"%"

        print str(label)+" | "+model

def avg_auc(dic_results):
    """ Average test AUC over total number of experiments """

    print "Label | Average AUC | Std Deviation\n"

    #loop across different classes and extract auc measures 
    avg_auc_list = []
    std_auc_list = []
    for label in dic_results.keys():
        auc = [float(r[2]) for r in dic_results[label].values() if not np.isnan(r[2])]

        #correct AUC values under 0.5
        for i in range(len(auc)):
            if auc[i] < 0.5: auc[i] = 1 - auc[i]

        #estimate AUC average and std dev
        auc_mean = np.average(np.array(auc))
        auc_std = np.std(np.array(auc))
        print str(label)+" | "+str(auc_mean)+" | "+str(auc_std)
        avg_auc_list.append(auc_mean)
        std_auc_list.append(auc_std)

    return avg_auc_list,std_auc_list

def hist_auc_average(auc_list,files):
    """ """
    N = 5 
    ind = np.arange(N)
    width = 0.4

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects = []
    colors=['r','y','g','c','b']
    i=0
    for model in auc_list:
        rects.append(ax.bar(ind+i*width,np.average(model[0]),width,color=colors[i],yerr=np.average(model[1])))
        i+=1
    ax.set_ylabel('Average AUC')
    ax.set_xlabel('Set of Features')
    ax.set_title('Average AUC On Detection Per Set Of Features')
    ax.grid()
    ax.set_xticks(ind+width)
    ax.set_xticklabels([str(i) for i in range(1,N+1)])

    leg = ax.legend( [ r[0] for r in rects] , files)
    for t in leg.get_texts():
        t.set_fontsize('small')

    #for r in rects:
    #    autolabel(r,ax)
    plt.show()

def hist_auc(auc_list,files):
    """ """
    N = len(auc_list[0][0])
    ind = np.arange(N)
    width = 0.13

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects = []
    colors=['r','y','g','c','b']
    i=0
    for model in auc_list:
        rects.append(ax.bar(ind+i*width,model[0],width,color=colors[i],yerr=model[1]))
        i+=1
    ax.set_ylabel('Average AUC')
    ax.set_xlabel('User')
    ax.set_title('Average AUC Per User And Set Of Features')
    ax.grid()
    ax.set_xticks(ind+width)
    ax.set_xticklabels([str(i) for i in range(1,N+1)])

    leg = ax.legend( [ r[0] for r in rects] , files)
    for t in leg.get_texts():
        t.set_fontsize('small')

    #for r in rects:
    #    autolabel(r,ax)
    plt.show()

def autolabel(rects,ax):
    """ """

    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%float(height),
                ha='center', va='bottom')

def read_scores_file(file):
    """ """

    f = gzip.open(file, "rb")
    lines = f.read().split("\n")
    f.close()

    info_line = []
    labels_a = []
    y_a = []
    scores_a = []
    y = ""
    scores = ""
    i=0
    for l in lines:
        if i == 0:
            info_line.append(l)
        elif i == 1:
            labels = l.replace("[","").replace("]","")
            labels = [float(x) for x in labels.split(",")]
            labels_a.append(labels)
        elif i == 2:
            y = l.replace("[","").replace("]","")
            y = [float(x) for x in y.split(",")]
            y_a.append(y)
        elif i == 3:
            scores = l.replace("[","").replace("]","")
            scores = [float(x) for x in scores.split(",")]
            scores_a.append(scores)
        i += 1
        if i == 4:
            i = 0
    
    scores_data = [info_line[:-1], labels_a, y_a, scores_a]
    return scores_data

def calculate_scores(data, w, file):
    """ Read data from .palms.scores. Print info lines, new AUC from
        aggregated scores (number of agg. samples w) and average AUC 
        from all experiments.
    """

    auc = 0 
    auc_a = []
    rocs = []
    for l in range(len(data[0])):
        scores = aggregate(data[3][l], data[1][l], w)
        rocs.append((scores, data[2][l]))
        #print data[0][l]
        #fpc, tpc, auc = calculate_auc(scores, data[2][l])
        #print auc
        #auc_a.append(auc)

    #The vertical averaged ROC of all experiments is saved
    fpr, tpr, auc = roc_VA(rocs)
    print "Average AUC: ", auc
    f  = open(file[:file.rindex(".scores")]+".roc_"+str(w),"wb")
    f.write(str(fpr)+"\n")
    f.write(str(tpr))
    f.close()

def calculate_auc(scores, labels):
    """ Preprocessing of binary labels and call to function roc to
        compute the ROC and the AUC
    """

    new_labels = []
    for l in labels:
        if l == -1:
            l == 0
        new_labels.append(l)
    fpc, tpc, auc = roc(scores, new_labels)

    return fpc, tpc, auc

def aggregate(scores, labels, w):
    """ Compute the aggregated version of a set of libSVM prediction scores
        according to window size w and original class of the samples
    """

    l_s = zip(labels, scores)
    scores_s = []
    scores_a = []
    for i in range(len(l_s)):
        label = l_s[i][0]
        for s in range(w):
            try:
                if l_s[i+s][0] == label:
                    scores_s.append(l_s[i+s][1])
            except Exception as exc:
                break 
        scores_a.append(np.median(scores_s))
        scores_s = []
    return scores_a

def plot_rocs(files):
    """ Plot all ROCs from several .palms.roc_w files """
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors=["royalblue","indianred","goldenrod","darkgreen","violet"]
    i=0
    for f in files:
        fd = open(f,'r').read().replace("[","").split("]\n")
        fpr = fd[0][1:].replace(" ",",").split(",")
        tpr = fd[1][1:].replace("]","").replace(" ",",").split(",")
        fpr = [float(x) for x in fpr if x is not '']
        tpr = [float(x) for x in tpr if x is not '']
        ax.plot(fpr,tpr,colors[i],label="w = "+f[f.rindex("_")+1:],linewidth=3)
        i += 1
    handles, labels = ax.get_legend_handles_labels()
    hl = sorted(zip(handles, labels),
                key=operator.itemgetter(1))
    handles2, labels2 = zip(*hl)
    leg = ax.legend(handles2, labels2,
                    title="Aggregation window of decision\nvalues from SVM classifier",
                    loc=4)
    leg.get_title().set_fontsize('small')
    for t in leg.get_texts():
        t.set_fontsize('small')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_title('ROC')
    ax.grid()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    plt.show()


if __name__ == "__main__":
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("-f", "--file", dest="libsvm_file",
                      default="-",
                      help="LibSVM file (default %default)")
    parser.add_option("-o", "--out", dest="palms_file",
                      default="-",
                      help="PALMS output file (default %default)")
    parser.add_option("-m", "--mode", type="int", dest="mode",
                      default=0,
                      help="Data partition mode (0 basic, 1 advanced) (default %default)")
    parser.add_option("-g", "--aggregation", type="int", dest="aggregation_size",
                      default=0,
                      help="Window size for SVM scores aggregation mode (default %default)")
    parser.add_option("-b", "--best", dest="best_palms_file",
                      default="-",
                      help="PALMS file for best AUC model averaging (default %default)")
    parser.add_option("-a", "--avg", dest="avg_palms_file",
                      default="-",
                      help="PALMS file for model AUC averaging (default %default)")
    parser.add_option("-c", "--comp", dest="comp_palms_files",
                      default="-",
                      help="PALMS files for model AUC averaging comparison, separated by commas (,) (default %default)")
    parser.add_option("-s", "--scores", dest="scores_palms_file",
                      default="-",
                      help="PALM file for model AUC averaging comparison using scores (default %default)")
    parser.add_option("-r", "--rocs", dest="roc_palms_files",
                      default="-",
                      help="PALM files with ROC data for different SVM scores aggregation windows (default %default)")
    (options, args) = parser.parse_args()

    if options.libsvm_file == "-" and options.palms_file == "-" and options.avg_palms_file == "-" and options.comp_palms_files == "-" and options.scores_palms_file == "-" and options.roc_palms_files == "-":
        parser.print_help()
        sys.exit(1)

    if options.libsvm_file != "-" and options.palms_file != "-":
        dic_results = model_learning(options.libsvm_file, options.palms_file, options.mode)
        model_selection(dic_results)

    elif options.best_palms_file != "-":
        fd = open(options.best_palms_file,'r')
        nan = np.float64("NaN")
        dic_results = eval(fd.read())
        model_selection(dic_results)

    elif options.avg_palms_file != "-":
        fd = open(options.avg_palms_file,'r')
        nan = np.float64("NaN")
        dic_results = eval(fd.read())
        avg_auc(dic_results)

    elif options.comp_palms_files != "-":
        auc_data = []
        print options.comp_palms_files.split(",")
        for f in options.comp_palms_files.split(","):
            fd = open(f,'r')
            nan = np.float64("NaN")
            dic_results = eval(fd.read())
            auc_data.append(avg_auc(dic_results))
        #hist_auc(auc_data,sys.argv[2:])
        hist_auc_average(auc_data,options.comp_palms_files)
        
    elif options.scores_palms_file != "-":
        scores_data = read_scores_file(options.scores_palms_file)
        calculate_scores(scores_data,options.aggregation_size, options.scores_palms_file)

    elif options.roc_palms_files != "-":
        roc_files = options.roc_palms_files.split(",")
        print roc_files
        plot_rocs(roc_files)
