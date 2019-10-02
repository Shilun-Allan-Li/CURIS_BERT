#!/usr/bin/env python                                                                               
# -*- coding: utf-8 -*-                                                                             

from __future__ import absolute_import, division, print_function
import argparse
import pandas as pd
import os
import sys
import torch
import random
import numpy as np
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel
from conllu import parse
from io import open
from conllu import parse_incr


def joinParse(data_file):
    data = data_file.read()
    entireList = []
    partialList = []
    sentences = parse(data)
    for sentence in sentences:
        for word in sentence:
            partialList.append((word['form'], word['upostag'], sentence.metadata))
        entireList.append(partialList)
        partialList = []
    return entireList


class PosDataset(data.Dataset):
    '''                                                                                             
    Appends [CLS] and [SEP] token in the beginning and in the end                                   
    to conform to BERT convention, in addition to <pad>                                             
    '''
    def __init__(self, tagged):
        sents, tags_li, original_sentences = [], [], []
        for sent in tagged:
            words = [word_pos[0] for word_pos in sent]
            tags = [word_pos[1] for word_pos in sent]
            sentence = [word_pos[2] for word_pos in sent]
            sents.append(['[CLS]'] + words + ['[SEP]'])
            tags_li.append(['<pad>'] + tags + ['<pad>'])
            original_sentences.append(sentence)
        self.sents, self.tags_li, self.original_sentences = sents, tags_li, original_sentences

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags, sentences = self.sents[idx], self.tags_li[idx], self.original_sentences[idx]
        x, y = [], []
        is_heads = []
        for word, tag in zip(words, tags):
            tokens = tokenizer.tokenize(word) if word not in ('[CLS]', '[SEP]') else [word]
            tokenToId = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [-1] * (len(tokens) - 1)
            tag = [tag] + ['<pad>'] * (len(tokens) - 1)
            tagEach = [tag2idx[each] for each in tag]

            x.extend(tokenToId)
            is_heads.extend(is_head)
            y.extend(tagEach)

        assert len(x) == len(y) == len(is_heads), 'len(x) = {}, len(y) = {}, len(is_heads) = {}'.format(len(x), len(y), len(is_heads))
        seqlen = len(y)
        words = ' '.join(words)
        tags = ' '.join(tags)
        return words, x, is_heads, tags, y, seqlen, sentences

def pad(batch):
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-2)
    sentences = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]
    x = f(1, maxlen)
    f = lambda x, seqlen: [sample[x] + [-1] * (seqlen - len(sample[x])) for sample in batch]
    y = f(-3, maxlen)
    f = torch.LongTensor
    return words, f(x), is_heads, tags, f(y), seqlens, sentences


class Net(nn.Module):
    '''                                                                                                
    Code snippet with minimal modification from pytorch-pretrained-bert                                
    and PyTorch documentation                                                                          
    '''
    def __init__(self, vocab_size=None, device=None, case='case'):
        super().__init__()
        if args.large:
            self.bert = BertModel.from_pretrained('bert-large-cased') if case == 'case' else BertModel.from_pretrained('bert-large-uncased')
            self.fc = nn.Linear(1024, vocab_size)
        else:
            self.bert = BertModel.from_pretrained('bert-base-cased') if case == 'case' else BertModel.from_pretrained('bert-base-uncased')
            self.fc = nn.Linear(768, vocab_size)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.device = device

    def forward(self, x, y):
        x = x.to(device)
        y = y.to(device)
        self.bert.eval()
        with torch.no_grad():
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[args.layer]
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

def train(model, iterator, optimizer, criterion):
    '''                                                                                                
    Trains the model based on the step size of 20                                                      
    '''
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens, sentences = batch
        _y = y
        optimizer.zero_grad()
        logits, y, _ = model(x, y)
        
        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        
        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()
        if i % 20 == 0:
            print('step: {}, loss: {}'.format(i, loss.item()))

def eval(model, iterator, case, seed):
    '''                                                                                                
    Evaluates the model and prints the accuracy                                                        
    Also saves the outputs in the form of (form, gold_tag, pred_tag, sent)                             
    Differences between gold_tags and pred_tags can be captured by running a                           
    separate function `diff.py` with the specified arguments `--case` and `--seed`                     
    '''
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat, Sentences = [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens, sentences = batch
            _, _, y_hat = model(x, y)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            Sentences.extend(sentences)
            
    if args.large:
        save_file = "../out/result_large_{}_{}_{}".format(case, args.layer, seed)
    else:
        save_file = "../out/result_base_{}_{}_{}".format(case, args.layer, seed)
    with open(save_file, 'w') as fout:
        for words, is_heads, tags, y_hat, sentences in zip(Words, Is_heads, Tags, Y_hat, Sentences):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p, s in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1], sentences):
                fout.write('{} {} {} {}\n'.format(w, t, p, s['text']))

    y_true =  np.array([tag2idx[line.split()[1]] for line in open(save_file, 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split()[2]] for line in open(save_file, 'r').read().splitlines() if len(line) > 0])

    acc = (y_true == y_pred).astype(np.int32).sum() / len(y_true)

    print("Dev set accuracy = %.4f" % acc)
    
    
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data/UD_English-EWT", help="Input Data")
parser.add_argument("--case", type=str, default="case", help="Either 'case' or 'uncase'")
parser.add_argument("--seed", type=int, default=0, help="Seed for training set")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
parser.add_argument("--save","-s", type=str, default=None, help="save model name (in save folder)")
parser.add_argument("--layer", type=int)
parser.add_argument("--large", action='store_true')
args = parser.parse_args()
path = args.data_dir + '/'
case = args.case.lower()
if args.layer is None:
    args.layer = 16 if args.large else 7
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

train_data_file = open(path + 'en_ewt-ud-train.conllu', 'r', encoding='utf-8')
dev_data_file = open(path + 'en_ewt-ud-dev.conllu', 'r', encoding='utf-8')
trainPos = joinParse(train_data_file)
devPos = joinParse(dev_data_file)

tagsTrain = list(set(word_pos[1] for sent in trainPos for word_pos in sent))

tag2idx = {tag:idx for idx, tag in enumerate(tagsTrain)}
tag2idx['<pad>']=-1
idx2tag = {idx:tag for idx, tag in enumerate(tagsTrain)}
idx2tag[-1]='<pad>'
idx2tag[17]='<pad>'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.large:
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=False) if case == 'case' else BertTokenizer.from_pretrained('bert-large-uncased')
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False) if case == 'case' else BertTokenizer.from_pretrained('bert-base-uncased')
    
model = Net(vocab_size=len(tag2idx), device=device, case=case)
model.to(device)
model = nn.DataParallel(model)

train_dataset = PosDataset(trainPos)
dev_dataset = PosDataset(devPos)

for epoch in range(args.epochs):
    print("----beginning epoch #{}----".format(epoch))
    trainIteration = data.DataLoader(dataset=train_dataset,
                                     batch_size=10,
                                     shuffle=True,
                                     num_workers=1,
                                     collate_fn=pad,
                                     worker_init_fn = np.random.seed(seed + epoch))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate * (0.95**epoch))
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    train(model, trainIteration, optimizer, criterion)
if args.save is not None:
    torch.save(model, '../out/'+args.save)

devIteration = data.DataLoader(dataset=dev_dataset,
                               batch_size=10,
                               shuffle=False,
                               num_workers=1,
                               collate_fn=pad,
                               worker_init_fn = np.random.seed(seed))
eval(model, devIteration, case, seed)
