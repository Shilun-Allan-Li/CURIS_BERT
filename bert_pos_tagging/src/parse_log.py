# -*- coding: utf-8 -*-
"""
@author: Allan
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from xlwt import Workbook
from sklearn.metrics import confusion_matrix

class_name = ['NUM', 'X', 'PUNCT', 'PRON', 'INTJ', 'VERB', 'SYM', 'NOUN', 'ADV', 'ADP', 'ADJ', 'DET', 'PART', 'CCONJ', 'PROPN', 'SCONJ', 'AUX']
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          text=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize:cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    for i in range(cm.shape[0]):cm[i,i] = 0
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if text:
        #divide = 10
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                num = cm[i, j]#if normalize else cm[i, j]//divide
                ax.text(j, i, format(num, fmt) if i!=j else "-",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def write_sheet(name, word, y_true, y_pred, sentence):
    wb = Workbook() 
    sheet = wb.add_sheet('misclassified words') 
    sheet.write(0, 0, 'word') 
    sheet.write(0, 1, 'gold_tag')
    sheet.write(0, 2, 'predict')
    sheet.write(0, 3, 'sentence')
    row = 1
    for i in range(len(word)):
        if y_true[i]==y_pred[i]: continue
        sheet.write(row, 0, word[i])
        sheet.write(row, 1, y_true[i])
        sheet.write(row, 2, y_pred[i])
        sheet.write(row, 3, sentence[i])
        row+=1
    wb.save("../out/saves/"+name+"_misclass.xls")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", type=str)
    parser.add_argument("-e",action="store_true")
    args = parser.parse_args()
    
    name=args.file
    y_true = []
    y_pred = []
    word = []
    sentence = []
    for line in open("../out/" + name, 'r').read().splitlines():
        if len(line) == 0: continue
        word.append(line.split(' ',3)[0])
        y_true.append(line.split(' ',3)[1])
        y_pred.append(line.split(' ',3)[2])
        sentence.append(line.split(' ', 3)[3])
    if args.e:
        plot_confusion_matrix(y_true, y_pred, classes=class_name, normalize=False,
                          title='Confusion matrix for pos tagging', text=True)
        plt.savefig("../out/saves/" + name + "_confusion.png")
    
        write_sheet(name, word, y_true, y_pred, sentence)
    
    right = np.sum(np.array(y_true)==np.array(y_pred))
    total = len(y_true)
    accuracy = right / total
    print("the accuracy of the {} is {}({}/{})".format(args.file, accuracy,right,total))
    #print("{},".format(accuracy))
