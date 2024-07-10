import seaborn as sns
from sklearn import metrics
import os
import numpy as np

# from https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification
def true_positive(y_true, y_pred):
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    
    return tp

def true_negative(y_true, y_pred):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
            
    return tn

def false_positive(y_true, y_pred):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
            
    return fp

def false_negative(y_true, y_pred):
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
            
    return fn

#Computation of macro-averaged precision
def macro_precision(y_true, y_pred):

    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize precision to 0
    precision = 0
    
    # loop over all classes
    for class_ in list(np.unique(y_true)):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)
        
        # compute false positive for current class
        fp = false_positive(temp_true, temp_pred)
        
        # compute precision for current class
        temp_precision = tp / (tp + fp + 1e-6)
        # keep adding precision for all classes
        precision += temp_precision
        
    # calculate and return average precision over all classes
    precision /= num_classes
    
    return precision

#Computation of macro-averaged recall
def macro_recall(y_true, y_pred):

    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize recall to 0
    recall = 0
    
    # loop over all classes
    for class_ in list(np.unique(y_true)):
        
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)
        
        # compute false negative for current class
        fn = false_negative(temp_true, temp_pred)
        
        # compute recall for current class
        temp_recall = tp / (tp + fn + 1e-6)
        
        # keep adding recall for all classes
        recall += temp_recall
        
    # calculate and return average recall over all classes
    recall /= num_classes
    
    return recall

import matplotlib.pyplot as plt
def make_histogram(l, epoch, name, n_speakers=1, by_epoch=True, ckpt_folder = ''):
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    plt.figure()
    plt.hist(l, bins=n_speakers)
    plt.title(name)
    if not by_epoch:
        plt.savefig(
            os.path.join(ckpt_folder,
                f'{name}.png'), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(
            os.path.join(ckpt_folder,
                f'{name}_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def make_confusion(y_gt, y_pred, name, epoch, label_encoder, ckpt_folder=''):
    plt.figure()
    ticklabels = label_encoder.inverse_transform(np.unique(
        np.concatenate((np.array(y_gt), np.array(y_pred)))))

    sns.heatmap(metrics.confusion_matrix(y_gt, np.array(y_pred)), annot=True,
        xticklabels = ticklabels, yticklabels = ticklabels, cmap='summer')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(
        os.path.join(ckpt_folder,
            f'{name}_{epoch}_confusion.png'), dpi=300, bbox_inches='tight')
    plt.close()