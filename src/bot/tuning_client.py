
# -*- coding: utf-8 -*-
'''
Created on : Monday 06 Apr, 2020 : 04:17:26
Last Modified : Saturday 20 Jun, 2020 : 01:19:28

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
import GPUtil as gputil
import subprocess
try:
	from subprocess import DEVNULL
except:
	import os
	DEVNULL = open(os.devnull, 'wb')

def get_cmd(q):
	config = q.dequeServer()
	if config == -1:
		print ('All Jobs Over!!!')
		exit(0)
	
	cmd = 'python train.py '
	for key, value in config.items():
		if key != 'if_self_loops':
			cmd += ' -'+key+' '+str(value)
		else:
			cmd += ' -self_loops'
	cmd += ' -data /projects/tir1/users/rjoshi2/negotiation/negotiation_personality/data/negotiation_data/data/strategy_vector_data_FULL_Yiheng.pkl'
	return cmd

def gpu_run(q, exclude, thresh):
	while True:
		gpus = gputil.getAvailable(order = 'memory', limit = 6, maxMemory = thresh,
				excludeID = exclude, excludeUUID=[])
		if len(gpus) == 0:
			time.sleep(10)
			continue
		for gpu in gpus:
			cmd = get_cmd(q)
			cmd += ' -gpu '+str(gpu)
			#cmd = 'source /usr1/home/rjoshi2/envs/myenv/bin/activate;' + cmd
			print ('Command : {}'.format(cmd))
			os.system(cmd)
			# my_env = os.environ.copy()
			# my_env['PATH'] = '/usr/sbin:/sbin:' + my_env['PATH']
			# pipe = subprocess.Popen(cmd.split(), stdin=None, stdout=None, stderr=None,env=my_env)
		time.sleep(60)

def cpu_run(q):
	while True:
		#cmd = '. /usr1/home/rjoshi2/envs/myenv/bin/activate;' + get_cmd(q) + ' -gpu -1'
		cmd = get_cmd(q) + ' -gpu -1'
		print ('Command: {}'.format(cmd))
		os.system(cmd)
		time.sleep(60)
		# my_env = os.environ.copy()
		# my_env['PATH'] = '/usr/sbin:/sbin:' + my_env['PATH']
		# pipe = subprocess.Popen(cmd.split(), stdin=None, stdout=None, stderr=None,env=my_env)
		# time.sleep(900)

if __name__ == '__main__':
	q = QueueClient('http://128.2.205.19:7979/')
	parser = argparse.ArgumentParser(description = 'Model Tuner')
	parser.add_argument('-gpu', default = '0')
	parser.add_argument('-cpu', action='store_true')
	parser.add_argument('-thresh', type=float, default = 0.8)
	args = parser.parse_args()
	if args.cpu:
		cpu_run(q)
	else:
		g = [int(w) for w in args.gpu.strip().split(',')]
		exclude = list(set([0,1,2,3,4,5,6,7]) - set(g))
		gpu_run(q, exclude, args.thresh)
