
# -*- coding: utf-8 -*-
'''
Created on : Monday 06 Apr, 2020 : 02:55:42
Last Modified : Friday 19 Jun, 2020 : 23:21:54

@author       : Rishabh Joshi
Institute     : Carnegie Mellon University
'''
import os, sys, pdb, numpy as np, random, argparse, codecs, pickle, time, json
from pprint import pprint
from collections import defaultdict as ddict
from joblib import Parallel, delayed
from pymongo import MongoClient

sys.path.append("/home/rjoshi2/dotfiles/utils/queue/")
from queue_client import QueueClient

q = QueueClient('http://128.2.205.19:7979/')

exclude_ids = set()
q.clear()
print ('Queue Cleared')

# LSTM
_ratio = ['0.3', '0.5', '0.8']
_graph_hidden = ['32', '64', '128']
_graph_layers = ['1', '3']
_graph_drop = ['0.0', '0.2']
_if_self_loops = [True]#, False]
_lr = ['0.001', '0.005', '0.0005']
_max_num_utt = ['128', '64']

_data = '/usr1/home/rjoshi2/negotiation_personality/data/negotiation_data/data/strategy_vector_data_FULL_Yiheng.pkl'

i, count = 0, 0
for sl in _if_self_loops:
	for drop in _graph_drop:
		for layer in _graph_layers:
			for hid in _graph_hidden:
				for r in _ratio:
					for lr in _lr:
						for mnu in _max_num_utt:
							config = {
								'name': 'run_{}'.format(i),
								'model': 'basic',
								'num_workers': '10',
								'strat_model': 'graph',
								'max_num_utt': mnu,
								'lr': lr,
								'ratio': r,
								'if_self_loops': sl,
								'graph_hidden' : hid,
								'graph_layers': layer,
								'graph_drop': drop,
								'data': _data}
							if i not in exclude_ids:
								count += 1
								q.enqueue(config)
								print ('Inserting {}'.format(count), end = '\r')
								# if i in [12, 13]:
								# 	q.enqueue(config)
								i += 1
print ('\nInserted {}, Total {} in queue. Complete'.format(count, q.getSize()))
