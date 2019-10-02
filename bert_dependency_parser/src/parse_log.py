# -*- coding: utf-8 -*-
"""
@author: Allan
"""
import argparse
import numpy as np
from xlwt import Workbook

def write_sheet(name, word, y_true, y_pred, sentence):
    wb = Workbook() 
    sheet = wb.add_sheet('misparsed words') 
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
    wb.save("../out/"+name+"_misparse.xls")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", type=str)
    parser.add_argument("-e","--eval", action='store_true')
    args = parser.parse_args()
    
    name=args.file
    y_true, y_pred, word, sentence = [], [], [], []
    for line in open("../out/" + name, 'r').read().splitlines():
        if len(line) == 0: continue
        word.append(line.split(' ',3)[0])
        y_true.append(line.split(' ',3)[1])
        y_pred.append(line.split(' ',3)[2])
        sentence.append(line.split(' ', 3)[3])
        
    if args.eval:
        write_sheet(name, word, y_true, y_pred, sentence)
    
    right = np.sum(np.array(y_true)==np.array(y_pred))
    total = len(y_true)
    accuracy = right / total
    print("the accuracy of {} is {}({}/{})".format(name, accuracy, right, total))
