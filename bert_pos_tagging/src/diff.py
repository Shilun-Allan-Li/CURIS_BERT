# -*- coding: utf-8 -*-
"""
@author: Allan
"""
import sys
import argparse
import numpy as np
from xlwt import Workbook
from collections import Counter
from pytorch_pretrained_bert import BertTokenizer

def write_sheet(name, rows):
    wb = Workbook() 
    sheet = wb.add_sheet('misclassified words') 
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            sheet.write(i, j, rows[i][j])
    wb.save("../out/saves/"+name+"_diff.xls")

if __name__=="__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", type=str)
    parser.add_argument("-f2","--file2", type=str)
    parser.add_argument("-r","--range", type=int)
    args = parser.parse_args()   
    
    word = [line.split(' ',3)[0] for line in open("../out/" + args.file+"_0", 'r').read().splitlines()]
    y_true = np.array([line.split(' ',3)[1] for line in open("../out/" + args.file+"_0", 'r').read().splitlines()])
    y_pred_all = np.array([[line.split(' ',3)[2] for line in open("../out/" + args.file+"_{}".format(i), 'r').read().splitlines()] for i in range(args.range)]).T
    y_pred_all2 = np.array([[line.split(' ',3)[2] for line in open("../out/" + args.file2+"_{}".format(i), 'r').read().splitlines()] for i in range(args.range)]).T
    sentence = [line.split(' ',3)[3] for line in open("../out/" + args.file+"_0", 'r').read().splitlines()]
    total = len(word)
    assert len(word) == len(sentence) == len(y_true) == len(y_pred_all) == len(y_pred_all2)
    rows = [[args.file + ' accuracy', args.file2 + ' accuracy', 'word', 'wordpiece', args.file, args.file2, 'gold_tag', 'sentence']]
    for i in range(len(y_pred_all)):
        preds = y_pred_all[i]
        preds2 = y_pred_all2[i]
        wrong = preds != y_true[i]
        wrong2 = preds2 != y_true[i]
        if abs(np.sum(wrong) - np.sum(wrong2)) > 0.6 * args.range:
            file1_pred, file2_pred = "", ""
            for c in Counter(preds[wrong]):
                file1_pred += "{}: {}, ".format(c, Counter(preds[wrong])[c])
            for c in Counter(preds2[wrong2]):
                file2_pred += "{}: {}, ".format(c, Counter(preds2[wrong2])[c])
            tokens = tokenizer.tokenize(word[i])
            rows.append([np.sum(wrong)/args.range, np.sum(wrong2)/args.range, word[i], str(tokens), file1_pred, file2_pred, y_true[i], sentence[i]])
    write_sheet(args.file + "_" + args.file2 ,rows)
