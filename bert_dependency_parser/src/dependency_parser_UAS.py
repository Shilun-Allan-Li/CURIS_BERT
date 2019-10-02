#!/usr/bin/env python                                                                               
# -*- coding: utf-8 -*-
"""
@author: Allan
"""                                                                            

from __future__ import absolute_import, division, print_function
import argparse
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
from chuliu_edmonds import chuliu_edmonds_one_root

def joinParse(data_file):
    data = data_file.read()
    entireList = []
    partialList = []
    sentences = parse(data)
    for sentence in sentences:
        for word in sentence:
            try:
                int(word['head'])
                partialList.append((word['form'], word['head'], sentence.metadata))
            except:
                continue
        entireList.append(partialList)
        partialList = []
    return entireList


class PosDataset(data.Dataset):
    '''                                                                                             
    Appends [CLS] and [SEP] token in the beginning and in the end                                   
    to conform to BERT convention, in addition to <pad>                                             
    '''
    def __init__(self, all_sents):
        sents, heads_li, original_sentences = [], [], []
        for sent in all_sents:
            words = [word_pos[0] for word_pos in sent]
            heads = [word_pos[1] for word_pos in sent]
            sentence = [word_pos[2] for word_pos in sent]
            sents.append(['[CLS]'] + words + ['[SEP]'])
            heads_li.append(['<pad>'] + heads + ['<pad>'])
            original_sentences.append(sentence)
        self.sents, self.heads_li, self.original_sentences = sents, heads_li, original_sentences

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, heads, sentences = self.sents[idx], self.heads_li[idx], self.original_sentences[idx]
        x, y = [], []
        is_heads = []
        head2idx = {0:0}
        idx2word = {0:'root'}
        index = 0
        head_index = 1
        for word in words:
            if word in ('[CLS]', '[SEP]'):
                index += 1
            else:
                tokens = tokenizer.tokenize(word)
                for i in range(len(tokens)):
                    idx2word[index+i] = word
                head2idx[head_index] = index
                index += len(tokens)
                head_index += 1
        
        for word, head in zip(words, heads):
            tokens = tokenizer.tokenize(word) if word not in ('[CLS]', '[SEP]') else [word]
            tokenToId = tokenizer.convert_tokens_to_ids(tokens)
            
            is_head = [1] + [0] * (len(tokens) - 1)
            head = [head] + ['<pad>'] * (len(tokens) - 1)
            headEach = [-1 if each=="<pad>" else head2idx[int(each)] for each in head]

            x.extend(tokenToId)
            is_heads.extend(is_head)
            y.extend(headEach)

        assert len(x) == len(y) == len(is_heads), 'len(x) = {}, len(y) = {}, len(is_heads) = {}'.format(len(x), len(y), len(is_heads))
        seqlen = len(y)
        words = ' '.join(words)
        heads = ' '.join([head if head=='<pad>' else str(head2idx[head]) for head in heads])
        return words, x, is_heads, heads, y, seqlen, sentences, idx2word

def pad(batch):
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    heads = f(3)
    seqlens = f(5)
    sentences = f(6)
    idx2word = f(7)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]
    x = f(1, maxlen)
    is_heads = f(2, maxlen)
    f = lambda x, seqlen: [sample[x] + [-1] * (seqlen - len(sample[x])) for sample in batch]
    y = f(4, maxlen)
    f = torch.LongTensor
    assert len(words) == len(sentences)
    return words, f(x), is_heads, heads, f(y), seqlens, sentences, idx2word


class Net(nn.Module):
    '''                                                                                                
    Code snippet with minimal modification from pytorch-pretrained-bert                                
    and PyTorch documentation                                                                          
    '''
    def __init__(self, hidden_dim=None, device=None, case='case'):
        super().__init__()
        if args.large:
            self.bert = BertModel.from_pretrained('bert-large-cased') if case == 'case' else BertModel.from_pretrained('bert-large-uncased')
            self.L1 = nn.Linear(1024, hidden_dim)#for head vectors
            self.L2 = nn.Linear(1024, hidden_dim)#for dependence vectors
        else:
            self.bert = BertModel.from_pretrained('bert-base-cased') if case == 'case' else BertModel.from_pretrained('bert-base-uncased')
            self.L1 = nn.Linear(768, hidden_dim)#for head vectors
            self.L2 = nn.Linear(768, hidden_dim)#for dependence vectors
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
        heads = self.L1(enc)
        dependence = self.L2(enc)
        heads_T = torch.transpose(heads, 1, 2)
        score = torch.matmul(dependence, heads_T)
        y_hat = score.argmax(-1)
        return score, y, y_hat

def train(model, iterator, optimizer, criterion):
    '''                                                                                                
    Trains the model based on the step size of 20                                                      
    '''
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, heads, y, seqlens, sentences, idx2word = batch
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
    Also saves the outputs in the form of (form, gold_head, pred_head, sent)                             
    Differences between gold_heads and pred_heads can be captured by running a                           
    separate function `diff.py` with the specified arguments `--case` and `--seed`                     
    '''
    model.eval()

    Words, Is_heads, Heads, Y,Sentences, Idx2word, Scores = [], [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, heads, y, seqlens, sentences, idx2word = batch
            score, _, _ = model(x, y)

            Scores.extend(score.cpu().numpy().tolist())
            Words.extend(words)
            Is_heads.extend(is_heads)
            Heads.extend(heads)
            Y.extend(y.numpy().tolist())
            Sentences.extend(sentences)
            Idx2word.extend(idx2word)   

    if args.large:
        save_file = "../out/UAS_large_{}_{}_{}_{}".format(case, args.hidden_dim, args.layer, seed)
    else:
        save_file = "../out/UAS_base_{}_{}_{}_{}".format(case, args.hidden_dim, args.layer, seed)
    print(save_file)
    with open(save_file, 'w') as fout:
        for words, is_heads, heads, sentences, idx2word, score in zip(Words, Is_heads, Heads, Sentences, Idx2word, Scores):
            is_heads = np.array(is_heads)==1
            score = np.array(score)[is_heads]
            score = score[:,is_heads]
            preds = [words.split()[i] if words.split()[i]!='[CLS]' else 'root' for i in chuliu_edmonds_one_root(score)[:-1]]
            assert (len(preds) + 1) == len(words.split()) == len(heads.split()) == (len(sentences) + 2)
            for w, t, p, s in zip(words.split()[1:-1], heads.split()[1:-1], preds[1:], sentences):
                fout.write('{} {} {} {}\n'.format(w, idx2word[int(t)], p, s['text']))

    y_true =  np.array([line.split()[1] for line in open(save_file, 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([line.split()[2] for line in open(save_file, 'r').read().splitlines() if len(line) > 0])
    right = (y_true == y_pred).astype(np.int32).sum()
    total = len(y_true)

    print("Dev set accuracy = %.4f" % (right/total) + "({}/{})".format(right, total))
    
def eval_accuracy(model, iterator, case, seed):
    model.eval()
    Words, Is_heads, Heads, Y,Sentences, Idx2word, Scores = [], [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, heads, y, seqlens, sentences, idx2word = batch
            score, _, _ = model(x, y)

            Scores.extend(score.cpu().numpy().tolist())
            Words.extend(words)
            Is_heads.extend(is_heads)
            Heads.extend(heads)
            Y.extend(y.numpy().tolist())
            Sentences.extend(sentences)
            Idx2word.extend(idx2word)   
    y_pred, y_true = [], []
    for words, is_heads, heads, sentences, idx2word, score in zip(Words, Is_heads, Heads, Sentences, Idx2word, Scores):
        is_heads = np.array(is_heads)==1
        score = np.array(score)[is_heads]
        score = score[:,is_heads]
        preds = [words.split()[i] if words.split()[i]!='[CLS]' else 'root' for i in chuliu_edmonds_one_root(score)[:-1]]
        y_pred += preds[1:]
        y_true += list(map(lambda x : idx2word[int(x)], heads.split()[1:-1]))
    right = (np.array(y_true) == np.array(y_pred)).astype(np.int32).sum()
    total = len(y_true)

    print("Dev set accuracy = %.4f" % (right/total) + "({}/{})".format(right, total))

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data/UD_English-EWT", help="Input Data")
parser.add_argument("--case", type=str, default="case", help="Either 'case' or 'uncase'")
parser.add_argument("--seed", type=int, default=0, help="Seed for training set")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dim")
parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("--load","-l", type=str, default=None, help="Load model name (in save folder)")
parser.add_argument("--save","-s", type=str, default=None, help="save model name (in save folder)")
parser.add_argument("--large", action='store_true')
parser.add_argument("--layer", type=int)
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.large:
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=False) if case == 'case' else BertTokenizer.from_pretrained('bert-large-uncased')
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False) if case == 'case' else BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = PosDataset(trainPos)
dev_dataset = PosDataset(devPos)

devIteration = data.DataLoader(dataset=dev_dataset,
                               batch_size=10,
                               shuffle=False,
                               num_workers=1,
                               collate_fn=pad,
                               worker_init_fn = np.random.seed(seed))

if args.load is None:
    model = Net(hidden_dim=args.hidden_dim, device=device, case=case)
    model.to(device)
    model = nn.DataParallel(model)
    for epoch in range(args.epochs):
        print("----beginning epoch #{}----".format(epoch))
        print("------trining with params: lr={}, hidden_dim={}--------".format(args.learning_rate * (0.95**epoch), args.hidden_dim))
        trainIteration = data.DataLoader(dataset=train_dataset,
                                     batch_size=10,
                                     shuffle=True,
                                     num_workers=1,
                                     collate_fn=pad,
                                     worker_init_fn = np.random.seed(seed+epoch))
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate * (0.95**epoch))
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        train(model, trainIteration, optimizer, criterion)
        if epoch % 2 == 0:
#            eval_accuracy(model, trainIteration, case, seed)
            eval_accuracy(model, devIteration, case, seed)
    if args.save is not None:
        torch.save(model, '../out/'+args.save)
else:
    model = torch.load('../out/'+args.load)
eval(model, devIteration, case, seed)
#eval(model, trainIteration, case, seed)
