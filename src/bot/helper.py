
# -*- coding: utf-8 -*-
'''
Created on : Wednesday 01 Apr, 2020 : 01:49:35
Last Modified : Monday 29 Jun, 2020 : 06:24:46

@author       : Rishabh Joshi
Institute     : Carnegie Mellon University
'''
import os, sys, pdb, numpy as np, random, argparse, codecs, pickle, time, json, csv, copy, uuid, math
import logging, logging.config, itertools, pathlib, socket
from types import SimpleNamespace
from typing import Any, Union, List, Tuple, Optional, Iterator
import gensim

from tqdm import tqdm
from pprint import pprint
from pymongo import MongoClient
from collections import OrderedDict
from glob import glob
import warnings, re
from sklearn.exceptions import UndefinedMetricWarning
from joblib import Parallel, delayed
from collections import defaultdict as ddict, Counter
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.special import softmax
#from oauth2client.service_account import ServiceAccountCredentials

# Pytorch related imports
import torch, torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter as Param
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, RandomSampler, SequentialSampler
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from transformers import BertModel, BertTokenizer


def make_dir(dirpath):
	if not os.path.exists(dirpath):
		os.makedirs(dirpath)

def checkFile(filename):
	return pathlib.Path(filename).is_file()

def str_proc(x):
	return str(x).strip().lower()

def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)
	
def partition(lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def getChunks(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))

def mergeListInDict(list_of_dict_of_list):

	dict_of_lists = {key: [] for key, values in list_of_dict_of_list[0].items()}

	for itm in list_of_dict_of_list:
		for key, values in itm.items():
			dict_of_lists[key].append(values)

	return dict_of_lists


def get_logger(name, log_dir, config_dir):
	config_dict = json.load(open('{}/log_config.json'.format(config_dir)))
	config_dict['handlers']['file_handler']['filename'] = '{}/{}'.format(log_dir, name.replace('/', '-'))
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def to_gpu(batch, dev):
	batch_gpu = {}
	for key, val in batch.items():
		if   key.startswith('_'):		batch_gpu[key] = val
		elif type(val) == type({1:1}): 	batch_gpu[key] = {k: v.to(dev) for k, v in batch[key].items()}
		else: 				batch_gpu[key] = val.to(dev)
	return batch_gpu

class ResultsMongo:
	def __init__(self, params, ip='brandy.lti.cs.cmu.edu', port=27017, db='dialog', username='vashishths', password='yogeshwara'):
		self.p		= params
		self.client	= MongoClient('mongodb://{}:{}/'.format(ip, port), username=username, password=password)
		self.db		= self.client[db][self.p.log_db]
		# self.db.update_one({'_id': self.p.name}, {'$set': {'Params': }}, upsert=True)

	def add_results(self, best_val, best_test, best_epoch, train_loss):
		try:
			self.db.update_one({'_id': self.p.name}, {
				'$set': {
					'best_epoch'	: best_epoch,
					'best_val'	: best_val,
					'best_test'	: best_test,
					'Params'	: vars(self.p)
				},
				'$push':{
					'train_loss'	: round(float(train_loss), 4),
					'all_val'	: best_val,
					'all_test'	: best_test,
				}
			}, upsert=True)
		except Exception as e:
			print('\nMongo Exception Cause: {}'.format(e.args[0]))

def read_csv(fname):
	with open(fname) as f:
		f.readline()
		for data in csv.reader(f):
			yield data

def mean_dict(acc):
	return {k: np.round(np.mean(v), 3) for k, v in acc.items()}

def get_param(shape):
	param = Parameter(torch.Tensor(*shape)); 	
	xavier_normal_(param.data)
	return param

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def dump_pickle(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))
	print ('Pickle Dumped {}'.format(fname))

def load_pickle(fname):
	return pickle.load(open(fname, 'rb'))

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def tune_thresholds(labels, logits, method = 'tune'):
	'''
	Takes labels and logits and tunes the thresholds using two methods
	methods are 'tune' or 'zc' #Zach Lipton
	Returns a list of tuples (thresh, f1) for each feature
	'''
	if method not in ['tune', 'zc']:
		print ('Tune method should be either tune or zc')
		sys.exit(1)
	from sklearn.metrics import f1_score, precision_recall_fscore_support, precision_score
	res = []
	logits = sigmoid(logits)

	num_labels = labels.shape[1]

	def tune_th(pid, feat):
		max_f1, th = 0, 0		# max f1 and its thresh
		if method == 'tune':
			ts_to_test = np.arange(0, 1, 0.001)
			for t in ts_to_test:
				scr  = f1_score(labels[:, feat], logits[:, feat] > t)
				if scr > max_f1:
					max_f1	= scr
					th	= t
		else:
			f1_half = f1_score(labels[:, feat], logits[:, feat] > 0.5)
			th = f1_half / 2
			max_f1 = f1_score(labels[:, feat], logits[:, feat] > th)

		return (th, max_f1)
		
	res = Parallel(n_jobs = 1)(delayed(tune_th)(lbl, lbl) for lbl in range(num_labels))
	return res


# Code used by Yiheng

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    
def normalizeString(s, tmp_dict, scenario, normalize_price=True):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)  # Add space around punctuation RJ

    # maybe i need to keep the number
    if normalize_price:
        if not tmp_dict["price"]:  # Price given for scene None RJ IDK WHY
            s = re.sub(r"[^a-zA-Z.!?<>]+", r" ", s)
        else:
            s = bin_price(s, tmp_dict["price"], scenario)  # Price is subbed
    else:
        s = re.sub(r"[^a-zA-Z.!?<>0-9]+", r" ", s)

    s = re.sub(r"\s+", r" ", s).strip()  # multiple spaces become 1
    return s

def bin_price(s, current_price, scenario):
    if scenario["kbs"][0]["personal"]["Role"] == "buyer":
        target = scenario["kbs"][0]["personal"]["Target"]
        price = scenario["kbs"][1]["personal"]["Target"]
    elif scenario["kbs"][0]["personal"]["Role"] == "seller":
        target = scenario["kbs"][1]["personal"]["Target"]
        price = scenario["kbs"][0]["personal"]["Target"]
    else:
        print("Role is not matched!")
        exit()

    nor_price = 1.0 * (current_price - target) / (price - target)
    if nor_price > 2.0:
        nor_price = "<price>_2.0"
    elif nor_price < -2.0:
        nor_price = "<price>_-2.0"
    else:
        nor_price = "<price>_" + str(round(nor_price, 1))

    s = re.sub(r"[^a-zA-Z.!?<>0-9]+", r" ", s)

    return re.sub(r"\d+,?\d+", " " + nor_price + " ", s)

recommendation2uniformstrategymapping = {'seller_neg_sentiment': 'neg_sentiment',
 'seller_pos_sentiment': 'pos_sentiment',
 'buyer_neg_sentiment': 'neg_sentiment',
 'buyer_pos_sentiment': 'pos_sentiment',
 'first_person_plural_count_seller': 'first_person_plural_count',
 'first_person_singular_count_seller': 'first_person_singular_count',
 'first_person_plural_count_buyer': 'first_person_plural_count',
 'first_person_singular_count_buyer': 'first_person_singular_count',
 'third_person_singular_seller': 'third_person_singular',
 'third_person_plural_seller': 'third_person_plural',
 'third_person_singular_buyer': 'third_person_singular',
 'third_person_plural_buyer': 'third_person_plural',
 'number_of_diff_dic_pos': 'number_of_diff_dic_pos',
 'number_of_diff_dic_neg': 'number_of_diff_dic_neg',
 'buyer_propose': 'propose',
 'seller_propose': 'propose',
 'hedge_count_seller': 'hedge_count',
 'hedge_count_buyer': 'hedge_count',
 'assertive_count_seller': 'assertive_count',
 'assertive_count_buyer': 'assertive_count',
 'factive_count_seller': 'factive_count',
 'factive_count_buyer': 'factive_count',
 'who_propose': 'who_propose',
 'seller_trade_in': 'trade_in',
 'personal_concern_seller': 'personal_concern',
 'sg_concern': 'sg_concern',
 'liwc_certainty': 'liwc_certainty',
 'liwc_informal': 'liwc_informal',
 'politeness_seller_please': 'politeness_please',
 'politeness_seller_gratitude': 'politeness_gratitude',
 'politeness_seller_please_s': 'politeness_please',
 'ap_des': 'ap_des',
 'ap_pata': 'ap_pata',
 'ap_infer': 'ap_infer',
 'family': 'family',
 'friend': 'friend',
 'politeness_buyer_please': 'politeness_please',
 'politeness_buyer_gratitude': 'politeness_gratitude',
 'politeness_buyer_please_s': 'politeness_please',
 'politeness_seller_greet': 'politeness_greet',
 'politeness_buyer_greet': 'politeness_greet',
 '<start>': '<start>'}
recommendation_feature_mapping = {"seller_neg_sentiment":0,"seller_pos_sentiment":1,
                                  "buyer_neg_sentiment":2,"buyer_pos_sentiment":3,
                                  "first_person_plural_count_seller":4,"first_person_singular_count_seller":5,
                                  "first_person_plural_count_buyer":6,"first_person_singular_count_buyer":7,
                                  "third_person_singular_seller":8,"third_person_plural_seller":9,
                                  "third_person_singular_buyer":10,"third_person_plural_buyer":11,
                                  "number_of_diff_dic_pos":12,"number_of_diff_dic_neg":13,
                                  "buyer_propose":14,"seller_propose":15,
                                  "hedge_count_seller":16,"hedge_count_buyer":17,
                                  "assertive_count_seller":18,"assertive_count_buyer":19,
                                  "factive_count_seller":20,"factive_count_buyer":21,
                                  "who_propose":22,"seller_trade_in":23,
                                  "personal_concern_seller":24,"sg_concern":25,
                                  "liwc_certainty":26,"liwc_informal":27,
                                  "politeness_seller_please":28,"politeness_seller_gratitude":29,
                                  "politeness_seller_please_s":30,
                                  "ap_des":31,"ap_pata":32,"ap_infer":33,
                                  "family":34,"friend":35,
                                  "politeness_buyer_please":36,"politeness_buyer_gratitude":37,
                                  "politeness_buyer_please_s":38,
                                  "politeness_seller_greet":39,"politeness_buyer_greet":40}
yihengid2recommendation_feature = {v:k for k,v in recommendation_feature_mapping.items()}

def index_to_word(uter_index, scenario, tokenizerORid2word, model = 'basic'):
    '''
    :param uter_index: utterance index
    :param scenario: scenario
    :param is_bert: flag for if uter_index is bert
    :return: result list of tokens
    '''
    if scenario["kbs"][0]["personal"]["Role"] == "buyer":
        target = scenario["kbs"][0]["personal"]["Target"]
        price = scenario["kbs"][1]["personal"]["Target"]
    elif scenario["kbs"][0]["personal"]["Role"] == "seller":
        target = scenario["kbs"][1]["personal"]["Target"]
        price = scenario["kbs"][0]["personal"]["Target"]
    else:
        print ("Role is not matched!")
        exit()

    result = list()
    decoded_out = list()
    if model == 'bert':
        result = tokenizerORid2word.convert_ids_to_tokens(uter_index)
    else:
        #pdb.set_trace()
        for i in uter_index:
            if tokenizerORid2word[i].startswith("<price>_"):
                result.append(str(float(tokenizerORid2word[i].replace("<price>_","").replace(",","")) * (price-target) + target))
                decoded_out.append(int(i))
            elif tokenizerORid2word[i] == '[SEP]':
                break
            else:
                result.append(tokenizerORid2word[i])
                decoded_out.append(int(i))
    return result, decoded_out
