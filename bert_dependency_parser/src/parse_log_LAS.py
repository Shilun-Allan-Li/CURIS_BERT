# -*- coding: utf-8 -*-
"""
@author: Allan
"""
import argparse
import numpy as np
from xlwt import Workbook
from conllu import parse

class_name = ['NUM', 'X', 'PUNCT', 'PRON', 'INTJ', 'VERB', 'SYM', 'NOUN', 'ADV', 'ADP', 'ADJ', 'DET', 'PART', 'CCONJ', 'PROPN', 'SCONJ', 'AUX'    ]
class_right = {c:0 for c in class_name}
class_total = {c:0 for c in class_name}
target_right = {c:0 for c in class_name}
target_total = {c:0 for c in class_name}
pair_right = {(c1, c2):0 for c1 in class_name for c2 in class_name}
pair_total = {(c1, c2):0 for c1 in class_name for c2 in class_name}

def parse_tags():
    dev_data_file = open('../data/UD_English-EWT/en_ewt-ud-dev.conllu', 'r', encoding='utf-8')
    data_f = dev_data_file.read()
    sentences = parse(data_f)
    tag, target_tag = [], []
    for sentence in sentences:
        for word in sentence:
            try:
                target = int(word['head'])
                target_tag.append(sentence[target - 1]['upostag'])
                tag.append(word['upostag'])
            except:
                continue
    return tag, target_tag

def write_sheet(name, word, y_true, y_pred, sentence, tags, target_tags):
    wb = Workbook() 
    sheet = wb.add_sheet('misparsed words') 
    sheet.write(0, 0, 'word') 
    sheet.write(0, 1, 'gold_tag')
    sheet.write(0, 2, 'predict')
    sheet.write(0, 3, 'upos')
    sheet.write(0, 4, 'target_upos')
    sheet.write(0, 5, 'sentence')
    row = 1
    for i in range(len(word)):
        if y_true[i]==y_pred[i]: continue
        sheet.write(row, 0, word[i])
        sheet.write(row, 1, y_true[i])
        sheet.write(row, 2, y_pred[i])
        sheet.write(row, 3, tags[i])
        sheet.write(row, 4, target_tags[i])
        sheet.write(row, 5, sentence[i])
        row+=1
    wb.save("../out/"+name+"_misparse.xls")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", type=str)
    parser.add_argument("-e","--eval", action='store_true')
    parser.add_argument("-s","--sheet", action='store_true')
    args = parser.parse_args()
    
    name=args.file
    y_true, y_pred, words, sentence, tags_true, tags_pred = [], [], [], [], [], []
    tags, target_tags = parse_tags()
    lines = open("../out/" + name, 'r').read().splitlines()
    assert len(lines) == len(tags) == len(target_tags)
    for i in range(len(lines)):
        line = lines[i]
        if len(line) == 0: continue
        word, true, pred, true_tag, pred_tag, sent = line.split(' ',5)
        words.append(word)
        y_true.append(true)
        y_pred.append(pred)
        sentence.append(sent)
        tags_true.append(true_tag)
        tags_pred.append(pred_tag)
        tag = tags[i]
        target_tag = target_tags[i]
        class_total[tag] += 1
        target_total[target_tag] += 1
        pair_total[(tag, target_tag)] += 1
        if true == pred:
            class_right[tag] += 1
            target_right[target_tag] += 1
            pair_right[(tag, target_tag)] += 1

    if args.sheet:
        write_sheet(name, words, y_true, y_pred, sentence, tags, target_tags)        
    if args.eval:
        for c in class_total:
            print("{}: {}({}/{})".format(c, round(class_right[c]/class_total[c],3), class_right[c], class_total[c]))
        print("\n")
        for c in target_total:
            print("{}: {}({}/{})".format(c, round(target_right[c]/target_total[c],3), target_right[c], target_total[c]))
        print("\n")
        
        print_dict = {}
        for c in pair_total:
            if pair_total[c] == 0: continue
            s = "{}->{}: {}({}/{})".format(c[0], c[1], round(pair_right[c]/pair_total[c],3), pair_right[c], pair_total[c])
            print_dict[pair_total[c]] = s
#            print(s)
        for key, value in sorted(print_dict.items(), key=lambda x: x[0], reverse=True): 
            print(value)
            
    right = np.sum((np.array(y_true)==np.array(y_pred)) * (np.array(tags_true)==np.array(tags_pred)))
    total = len(y_true)
    accuracy = right / total
    print("the accuracy of {} is:".format(name))
    print("head: {}({}/{})".format(np.sum(np.array(y_true)==np.array(y_pred))/total, np.sum(np.array(y_true)==np.array(y_pred)), total))
    print("tags: {}({}/{})".format(np.sum(np.array(tags_true)==np.array(tags_pred))/total, np.sum(np.array(tags_true)==np.array(tags_pred)), total))
    print("total: {}({}/{})".format(accuracy, right, total))
