import sys
import copy
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Process, Queue
#import time
#import datetime as dt
#from datetime import date

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, user_train_side1, user_train_side2, user_train_side3, usernum, itemnum,item_side, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        seq_side1 = np.zeros([maxlen], dtype=np.int32)
        seq_side2 = np.zeros([maxlen], dtype=np.int32)
        seq_side3 = np.zeros([maxlen], dtype=np.int32)
        
        pos = np.zeros([maxlen], dtype=np.int32)
        pos_side1 = np.zeros([maxlen], dtype=np.int32)
        pos_side2 = np.zeros([maxlen], dtype=np.int32)
        pos_side3 = np.zeros([maxlen], dtype=np.int32)

        neg = np.zeros([maxlen], dtype=np.int32)
        neg_side1 = np.zeros([maxlen], dtype=np.int32)
        neg_side2 = np.zeros([maxlen], dtype=np.int32)
        neg_side3 = np.zeros([maxlen], dtype=np.int32)
        
        nxt = user_train[user][-1]
        nxt_side1 = user_train_side1[user][-1]
        nxt_side2 = user_train_side2[user][-1]
        nxt_side3 = user_train_side3[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            seq_side1[idx] = item_side.loc[i]['side1']
            seq_side2[idx] = item_side.loc[i]['side2']
            seq_side3[idx] = item_side.loc[i]['side3']

            pos[idx] = nxt
            pos_side1[idx] = nxt_side1
            pos_side2[idx] = nxt_side2
            pos_side3[idx] = nxt_side3
            
            if nxt != 0:
              neg[idx] = random_neq(1, itemnum + 1, ts)
              neg_side1[idx] = item_side.loc[neg[idx]]['side1']
              neg_side2[idx] = item_side.loc[neg[idx]]['side2']
              neg_side3[idx] = item_side.loc[neg[idx]]['side3']
            nxt = i
            nxt_side1 = item_side.loc[i]['side1']
            nxt_side2 = item_side.loc[i]['side2']
            nxt_side3 = item_side.loc[i]['side3']

            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg
        , seq_side1, pos_side1, neg_side1
        , seq_side2, pos_side2, neg_side2
        , seq_side3, pos_side3, neg_side3)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, user_train_side1, user_train_side2, user_train_side3, usernum, itemnum, item_side, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,user_train_side1, user_train_side2, user_train_side3,
                                                      usernum,
                                                      itemnum,
                                                      item_side,                                                      
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):

    
    usernum = 0
    itemnum = 0
    user_train = {}
    user_train_side1 = {}
    user_train_side2 = {}
    user_train_side3 = {}

    user_valid = {}
    user_valid_side1 = {}
    user_valid_side2 = {}
    user_valid_side3 = {}
    
    user_test = {}
    user_test_side1 = {}
    user_test_side2 = {}
    user_test_side3 = {}

    f = pd.read_csv('data/%s.csv'% fname,encoding = 'cp949')
    f = f[['회원번호' , '책제목' ,'일자', '카테고리' , '작가' , '출판사']]
    f.rename(columns = {'카테고리':'side1', '작가':'side2', '출판사':'side3'}, inplace = True)
    f.fillna('미정', inplace=True)

    umap = {u: (i+1) for i, u in enumerate(set(f['회원번호']))}
    smap = {s: (i+1) for i, s in enumerate(set(f['책제목']))}

    side1map = {u: (i+1) for i, u in enumerate(set(f['side1']))} 
    side2map = {u: (i+1) for i, u in enumerate(set(f['side2']))}
    side3map = {u: (i+1) for i, u in enumerate(set(f['side3']))}

    f['회원번호'] = f['회원번호'].map(umap)
    f['책제목'] = f['책제목'].map(smap)

    f['side1'] = f['side1'].map(side1map)
    f['side2'] = f['side2'].map(side2map)
    f['side3'] = f['side3'].map(side3map)

    item_side = f.drop(['회원번호'],axis=1).drop_duplicates(['책제목'])
    item_side = item_side.set_index('책제목')
    User= f.groupby('회원번호')['책제목'].apply(list).reset_index()
    User_side1= f.groupby('회원번호')['side1'].apply(list).reset_index()
    User_side2= f.groupby('회원번호')['side2'].apply(list).reset_index()
    User_side3= f.groupby('회원번호')['side3'].apply(list).reset_index()

    User=User.set_index('회원번호').to_dict()['책제목']
    User_side1=User_side1.set_index('회원번호').to_dict()['side1']
    User_side2=User_side2.set_index('회원번호').to_dict()['side2']
    User_side3=User_side3.set_index('회원번호').to_dict()['side3']

    usernum=len(umap)
    itemnum=len(smap)
    side1num=len(side1map)
    side2num=len(side2map)
    side3num=len(side3map)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_train_side1[user] = User_side1[user]
            user_train_side2[user] = User_side2[user]
            user_train_side3[user] = User_side3[user]
            user_valid[user] = []
            user_valid_side1[user] = []
            user_valid_side2[user] = []
            user_valid_side3[user] = []
            
            user_test[user] = []
            user_test_side1[user] = []
            user_test_side2[user] = []
            user_test_side3[user] = []
            
        else:
            user_train[user] = User[user][:-2]
            user_train_side1[user] = User_side1[user][:-2]
            user_train_side2[user] = User_side2[user][:-2]
            user_train_side3[user] = User_side3[user][:-2]

            user_valid[user] = []
            user_valid_side1[user] = []
            user_valid_side2[user] = []
            user_valid_side3[user] = []

            user_valid[user].append(User[user][-2])
            user_valid_side1[user].append(User_side1[user][-2])
            user_valid_side2[user].append(User_side2[user][-2])
            user_valid_side3[user].append(User_side3[user][-2])

            user_test[user] = []
            user_test_side1[user] = []
            user_test_side2[user] = []
            user_test_side3[user] = []

            user_test[user].append(User[user][-1])
            user_test_side1[user].append(User_side1[user][-1])
            user_test_side2[user].append(User_side2[user][-1])
            user_test_side3[user].append(User_side3[user][-1])

    return [user_train, user_valid, user_test, usernum, itemnum,
    user_train_side1, user_valid_side1, user_test_side1, side1num,
    user_train_side2, user_valid_side2, user_test_side2, side2num,
    user_train_side3, user_valid_side3, user_test_side3, side3num, item_side]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum,
    train_side1, valid_side1, test_side1, side1num,
    train_side2, valid_side2, test_side2, side2num,
    train_side3, valid_side3, test_side3, side3num, item_side] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seq_side1 = np.zeros([args.maxlen], dtype=np.int32)
        seq_side2 = np.zeros([args.maxlen], dtype=np.int32)
        seq_side3 = np.zeros([args.maxlen], dtype=np.int32)

        idx = args.maxlen - 1
        seq[idx] = valid[u][0]

        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        idx = args.maxlen - 1
        seq_side1[idx] = valid_side1[u][0]
        idx -= 1
        for i in reversed(train_side1[u]):
            seq_side1[idx] = i
            idx -= 1
            if idx == -1: break
        idx = args.maxlen - 1
        seq_side2[idx] = valid_side2[u][0]
        idx -= 1
        for i in reversed(train_side2[u]):
            seq_side2[idx] = i
            idx -= 1
            if idx == -1: break
        idx = args.maxlen - 1
        seq_side3[idx] = valid_side3[u][0]
        idx -= 1
        for i in reversed(train_side3[u]):
            seq_side3[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        rated_side1 = set(train_side1[u])
        rated_side1.add(0)
        rated_side2 = set(train_side2[u])
        rated_side2.add(0)
        rated_side3 = set(train_side3[u])
        rated_side3.add(0)

        item_idx = [test[u][0]]
        item_side1_idx = [test_side1[u][0]]
        item_side2_idx = [test_side2[u][0]]
        item_side3_idx = [test_side3[u][0]]

        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            item_side1_idx.append(item_side.loc[t]['side1'])
            item_side2_idx.append(item_side.loc[t]['side2'])
            item_side3_idx.append(item_side.loc[t]['side3'])

        # predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]],
        # *[np.array(l) for l in [[u], [seq_side1], item_side1_idx]],
        # *[np.array(l) for l in [[u], [seq_side2], item_side2_idx]],
        # *[np.array(l) for l in [[u], [seq_side3], item_side3_idx]])
        # predictions = predictions[0] # - for 1st argsort DESC
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, [seq_side1], item_side1_idx,[seq_side2], item_side2_idx,[seq_side3], item_side3_idx]])
        predictions = predictions[0] # - for 1st argsort DESC
    

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum,
    train_side1, valid_side1, test_side1, side1num,
    train_side2, valid_side2, test_side2, side2num,
    train_side3, valid_side3, test_side3, side3num, item_side] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seq_side1 = np.zeros([args.maxlen], dtype=np.int32)
        seq_side2 = np.zeros([args.maxlen], dtype=np.int32)
        seq_side3 = np.zeros([args.maxlen], dtype=np.int32)

        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        idx = args.maxlen - 1
        for i in reversed(train_side1[u]):
            seq_side1[idx] = i
            idx -= 1
            if idx == -1: break

        idx = args.maxlen - 1
        for i in reversed(train_side2[u]):
            seq_side2[idx] = i
            idx -= 1
            if idx == -1: break
            
        idx = args.maxlen - 1
        for i in reversed(train_side3[u]):
            seq_side3[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        rated_side1 = set(train_side1[u])
        rated_side1.add(0)
        rated_side2 = set(train_side2[u])
        rated_side2.add(0)
        rated_side3 = set(train_side3[u])
        rated_side3.add(0)
        
        item_idx = [valid[u][0]]
        item_side1_idx = [valid_side1[u][0]]
        item_side2_idx = [valid_side2[u][0]]
        item_side3_idx = [valid_side3[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            item_side1_idx.append(item_side.loc[t]['side1'])
            item_side2_idx.append(item_side.loc[t]['side2'])
            item_side3_idx.append(item_side.loc[t]['side3'])

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, [seq_side1], item_side1_idx,[seq_side2], item_side2_idx,[seq_side3], item_side3_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user