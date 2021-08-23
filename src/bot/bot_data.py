
# -*- coding: utf-8 -*-
'''
Created on : Friday 22 May, 2020 : 18:48:02
Last Modified : Tuesday 30 Jun, 2020 : 22:27:52

@author       : Rishabh Joshi
Institute     : Carnegie Mellon University
'''

from helper import *
from types import SimpleNamespace

# Sampler to be used from dialogue_data
class NegotiationDataBatchSampler(Sampler):
	def __init__(self, dataset, max_num_utt_batch = 1024, drop_last = False, shuffle = True):
		self.dataset    = dataset
		self.num_samples = len(self.dataset)
		self.max_num_utt_batch = max_num_utt_batch
		self.drop_last = drop_last
		if shuffle:
			self.sampler = RandomSampler(np.arange(self.num_samples), replacement = False)
		else:
			self.sampler = SequentialSampler(np.arange(self.num_samples))

	def __iter__(self):
		''' 
		provides a way to iterate over the dataset
		'''
		#pdb.set_trace()
		batch = []
		num_utt_batch = 0				# this is the num of utt in batch
		for idx in self.sampler:
			conv_len = self.dataset[idx]['strategies_vec'].shape[0]
			num_utt_batch_new = num_utt_batch + conv_len
			if num_utt_batch_new > self.max_num_utt_batch:
				#print ('Yielding : ', batch, num_utt_batch)
				#yield (batch, num_utt_batch, max_utt_len_batch)
				#yield {'batch': batch, 'num_utt_batch': num_utt_batch, 'max_utt_len_batch': max_utt_len_batch}
				if len(batch) == 0: # no idx added and current data is bigger - means batch size doesnt support it
					continue   # skip the datapoint
				yield batch
				batch = []
				max_utt_len_batch = 0
				num_utt_batch = 0
			num_utt_batch += conv_len
			batch.append(idx)
			
		if len(batch) > 0 and not self.drop_last:
			#print ('Yielding : ', batch, num_utt_batch, max_utt_len_batch)
			#yield (batch, num_utt_batch, max_utt_len_batch)
			#yield {'batch': batch, 'num_utt_batch': num_utt_batch, 'max_utt_len_batch': max_utt_len_batch}
			yield batch

	def __len__(self):
		'''
		returns length of the returned iterators
		'''
		return self.num_samples

class NegotiationBotDataset(Dataset):
	def __init__(self, data, config):
		self.dataset = data
		self.return_graph_data = False#config.only_eval
		if self.return_graph_data:
			self.undirected = config.undirected
			self.self_loops = config.self_loops

		#self.dataset = [{'agent_list', 'utterance', 'strategies', 'strategies_vec'}]
		#self.elmo_dim = config.elmo_dim
		#self.num_char = config.num_char

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		#print ('loading data ', idx)
		return self.dataset[idx]
		# strategies_vec = self.dataset[idx]['strategies_vec']				# array of array [[0,0,0,1], [1, 0, 1,0]]
		# ratio_bucket	= self.dataset[idx]['ratio_bucket']
		# word_ids	= self.dataset[idx]['toks_space']
		# bert_ids	= self.dataset[idx]['toks_bert']
		# dial_acts	= self.dataset[idx]['dial_acts_vec']
		# uuid		= self.dataset[idx]['uuid']
		# text		= self.dataset[idx]['utterance']
		# return strategies_vec, ratio, word_ids, bert_ids, dial_acts, uuid, text

	def collate_fn(self, batch):
		# [ {}, {}]
		# [ (strategies_vec[0], ratio[0]), (strategies_vec[1], ratio[1]), ... ()]
		num_conv = len(batch)
		num_utt = np.max([len(b['utterance']) for b in batch])				# max num utt in any conv
		num_strategies = batch[0]['strategies_vec'].shape[1]				# 1 for clustered strategies
		max_word_seq = np.max([len(x) for b in batch for x in b['toks_space']])
		max_bert_seq = np.max([len(x) for b in batch for x in b['toks_bert']])
		#max_word_seq = np.max([len(b['toks_space']) for b in batch])
		#max_bert_seq = np.max([len(b['toks_bert']) for b in batch])

		strategy_seq	= np.zeros((num_conv, num_utt, num_strategies))
		utt_mask	= np.zeros((num_conv, num_utt))
		ratio_bucket	= np.zeros((num_conv, 1))
		word_input	= np.zeros((num_conv, num_utt, max_word_seq))
		word_mask	= np.zeros((num_conv, num_utt, max_word_seq))
		bert_input	= np.zeros((num_conv, num_utt, max_bert_seq))
		bert_mask	= np.zeros((num_conv, num_utt, max_bert_seq))
		da_input	= np.zeros((num_conv, num_utt, 1))
		agent_list	= np.full((num_conv, num_utt, 1), -1)
		uuids		= [b['uuid'] for b in batch]
		texts		= [b['utterance'] for b in batch]

		for i in range(num_conv):
			curr_utt_num	    = len(batch[i]['utterance'])
			for j in range(curr_utt_num):
				curr_toks_space_num = len(batch[i]['toks_space'][j])#.shape[1]
				curr_toks_bert_num  = len(batch[i]['toks_bert'][j])#.shape[1]
				word_input[i, j, 0:curr_toks_space_num]		= batch[i]['toks_space'][j]
				word_mask[i, j, 0:curr_toks_space_num]		= 1.0
				bert_input[i, j, 0:curr_toks_bert_num]		= batch[i]['toks_bert'][j]
				bert_mask[i, j, 0:curr_toks_bert_num]		= 1.0

			strategy_seq[i, 0:curr_utt_num, :]				= batch[i]['strategies_vec']
			utt_mask[i, 0:curr_utt_num]					= 1.0
			ratio_bucket[i][0]						= batch[i]['ratio_bucket']
			da_input[i, 0:curr_utt_num, 0]					= batch[i]['dial_acts_vec']
			agent_list[i, 0:curr_utt_num, 0]				= batch[i]['agent_list']

		if self.return_graph_data:
			from torch_geometric.data import Batch
			graph_list = []
			for conv in batch:
				graph_list += self.convert_strategyvec_to_graph(conv['strategies_vec'])		# List of torch_geometric.data Data

			batch_graph	= Batch.from_data_list(graph_list)


		batch = {
			#'feats'		: torch.FloatTensor(input_batch), 
			'ratio_bucket'	: torch.LongTensor(ratio_bucket), 
			'num_conv'	: torch.Tensor([num_conv]),
			'utt_mask'	: torch.FloatTensor(utt_mask),
			'strategy_seq'	: torch.FloatTensor(strategy_seq),
			'word_input'	: torch.LongTensor(word_input),
			'word_mask'	: torch.FloatTensor(word_mask),
			'bert_input'	: torch.LongTensor(bert_input),
			'bert_mask'	: torch.FloatTensor(bert_mask),
			'dial_act_input': torch.LongTensor(da_input),
			'agent_list'    : torch.LongTensor(agent_list),
			'uuids'		: uuids,					# Not torch type
			'texts'		: texts						# Not torch type
		}
		if self.return_graph_data:
			batch['input_graph'] = batch_graph
		#return torch.FloatTensor(input_batch), torch.LongTensor(utt_mask), torch.FloatTensor(ratios)
		return batch#, uuids, texts
	
	def convert_strategyvec_to_graph(self, strategies_vec):
		'''
		Takes a strategies vector and converts it to a list of torch_geometric.data Data items
		'''
		from torch_geometric.data import Data
		graph_data = []
		adj_x, adj_y = [], []
		# skip for time step 0
		# lower triangle useful
		total_rows = 0
		for i in range(len(strategies_vec)):
			#adj_y.append(np.array(strategies_vec[i+1]))
			num_strategies_in_turn	= int(np.sum(strategies_vec[i]))
			new_matrix		= np.zeros((total_rows + num_strategies_in_turn, total_rows + num_strategies_in_turn))
			new_strategies		= np.zeros((total_rows + num_strategies_in_turn, 1))
			if i != 0:
				new_matrix[: total_rows, : total_rows]	= adj_x[i-1]['matrix']	# copy prev matrix
				new_strategies[: total_rows]		= adj_x[i-1]['strategies']
			curr_row = total_rows
			for stidx, sval in enumerate(strategies_vec[i]):
				if sval == 0: continue
				new_strategies[curr_row, 0]	   = stidx
				#new_strategies.append(stidx)
				new_matrix[curr_row, : total_rows] = 1			# connecting to all in lower half except self
				curr_row += 1
			total_rows = curr_row
			adj_x.append({
				'matrix':	new_matrix,
				'strategies':	new_strategies
			})
			x		= torch.LongTensor(new_strategies)						# (num_strategies, 1) for now. Later will do embedding lookup
			edge_index	= self.get_edge_index_from_adj_matrix(torch.LongTensor(new_matrix))
			try:
				y = torch.FloatTensor(strategies_vec[i+1][:-1].reshape(1, -1))
			except:
				y = None
			#y		= torch.FloatTensor(np.array(strategies_vec[i+1]).reshape(1, -1))

			graph_data.append(Data(x = x, edge_index = edge_index, y = y))
		return graph_data

	def get_edge_index_from_adj_matrix(self, adj_matrix):
		from torch_geometric.utils.sparse import dense_to_sparse
		from torch_geometric.utils.undirected import to_undirected
		from torch_geometric.utils.loop import add_self_loops
		edge_index, edge_value = dense_to_sparse(adj_matrix)
		if edge_index.shape[1] != 0 and self.undirected:
			edge_index = to_undirected(edge_index)
		if edge_index.shape[1] != 0 and self.self_loops:
			edge_index, _ = add_self_loops(edge_index)
		return edge_index

class NegotiationGraphDataset(Dataset):
	def __init__(self, data, config):
		self.dataset = data

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		#print ('loading data ', idx)
		strategies_vec	= self.dataset[idx]['strategies_vec']				# array of array [[0,0,0,1], [1, 0, 1,0]]
		adj_x, adj_y	= self.convert_strategyvec_to_graph(strategies_vec)		# List of adj input and adj_output
		ratio		= self.dataset[idx]['ratio']
		return adj_x, adj_y, ratio

	def collate_fn(self, batch):
		# [ (adj_x[0], adj_y[0], ratio[0]), (adj_x[1], adj_y[1], ratio[1]), ... ()]
		num_conv	= len(batch)
		num_utt		= np.max([len(b[0]) for b in batch])				# max num utt in any conv
		num_graph_rows	= np.max([len(b[0][-1]['strategies']) for b in batch])	# max num of rows of any graph, see last turn
		num_strategies	= batch[0][1][0].shape[0]					# len of conv0 adj_y turn1

		input_graph		= np.zeros((num_conv, num_utt, num_graph_rows, num_graph_rows))
		input_strategies	= np.zeros((num_conv, num_utt, num_graph_rows, 1))
		graph_mask		= np.zeros((num_conv, num_utt, num_graph_rows))
		y			= np.zeros((num_conv, num_utt, num_strategies))
		utt_mask		= np.zeros((num_conv, num_utt))
		ratios			= np.zeros((num_conv, 1))

		for i in range(num_conv):
			for j in range(len(batch[i][0])):
				input_graph     [i, j, :batch[i][0][j]['matrix'].shape[0], :batch[i][0][j]['matrix'].shape[1]]	= batch[i][0][j]['matrix']
				input_strategies[i, j, :batch[i][0][j]['strategies'].shape[0]]					= batch[i][0][j]['strategies']
				graph_mask	[i, j, :batch[i][0][j]['strategies'].shape[0]]					= 1
			y		[i, :batch[i][1].shape[0],	     :batch[i][1].shape[1]]		= batch[i][1]
			utt_mask	[i, :batch[i][1].shape[0]]						= np.ones(batch[i][1].shape[0])
			ratios		[i][0]									= batch[i][2]

		batch = {
			'input_graph'		: torch.LongTensor(input_graph),
			'input_strategies'	: torch.LongTensor(input_strategies),
			'graph_mask'		: torch.LongTensor(graph_mask),
			'y'			: torch.FloatTensor(y),
			'ratios'		: torch.FloatTensor(ratios), 
			'utt_mask'		: torch.FloatTensor(utt_mask)
		}
		return batch

	def convert_strategyvec_to_graph(self, strategies_vec):
		'''
		Takes a strategies vector and converts it to a list of inputs and outputs
		adj_x is dict of {'matrix': , 'strategies': }
		'''
		adj_x, adj_y = [], []
		# skip for time step 0
		# lower triangle useful
		total_rows = 0
		for i in range(len(strategies_vec) - 1):
			adj_y.append(np.array(strategies_vec[i+1]))
			num_strategies_in_turn	= int(np.sum(strategies_vec[i]))
			new_matrix		= np.zeros((total_rows + num_strategies_in_turn, total_rows + num_strategies_in_turn))
			new_strategies		= np.zeros((total_rows + num_strategies_in_turn, 1))
			#new_strategies		= []
			if i != 0:
				new_matrix[: total_rows, : total_rows]	= adj_x[i-1]['matrix']	# copy prev matrix
				#new_strategies[: total_rows]		= adj_x[i-1]['strategies']
				new_strategies[: total_rows]		= adj_x[i-1]['strategies']
			curr_row = total_rows
			for stidx, sval in enumerate(strategies_vec[i]):
				if sval == 0: continue
				#new_strategies.append(stidx)
				new_strategies[curr_row, 0]	   = stidx
				new_matrix[curr_row, : total_rows] = 1			# connecting to all in lower half except self
				curr_row += 1
			total_rows = curr_row
			adj_x.append({
				'matrix':	new_matrix,
				'strategies':	new_strategies
			})
		return adj_x, np.array(adj_y)

class NegotiationGraphDatasetGeometric(Dataset):
	def __init__(self, data, config):
		self.dataset	= data
		self.undirected = config.undirected
		self.self_loops = config.self_loops

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		#print ('loading data ', idx)
		strategies_vec	= self.dataset[idx]['strategies_vec']				# array of array [[0,0,0,1], [1, 0, 1,0]]
		graph_data	= self.convert_strategyvec_to_graph(strategies_vec)		# List of torch_geometric.data Data
		ratio		= self.dataset[idx]['ratio']
		return graph_data, ratio, strategies_vec

	def collate_fn(self, batch):
		# [ (Data[0], ratio[0], strategies_vec[0]), (Data[1], ratio[1], strategies_vec[1]), ... ()]
		from torch_geometric.data import Batch
		graph_list = []
		for conv in batch:
			graph_list += conv[0]
		batch_graph	= Batch.from_data_list(graph_list)
		num_conv	= len(batch)
		num_utt		= np.max([b[2].shape[0] for b in batch])
		num_strategies  = batch[0][2].shape[1]

		input_batch	= np.zeros((num_conv, num_utt, num_strategies))
		utt_mask	= np.zeros((num_conv, num_utt))
		ratios		= np.zeros((num_conv, 1))

		for i in range(num_conv):
			input_batch[i, 0:batch[i][2].shape[0], :] = batch[i][2]
			utt_mask[i, 0:batch[i][2].shape[0]] = np.ones(batch[i][2].shape[0])
			ratios[i][0] = batch[i][1]

		batch = {
			'input_graph'		: batch_graph, 
			'ratios'		: torch.FloatTensor(ratios),			# TODO : IGNORE FOR NOW for Graphs
			'num_conv'		: torch.Tensor([num_conv]),
			'feats'			: torch.FloatTensor(input_batch), 
			'utt_mask'		: torch.FloatTensor(utt_mask)
		}
		return batch

	def convert_strategyvec_to_graph(self, strategies_vec):
		'''
		Takes a strategies vector and converts it to a list of torch_geometric.data Data items
		'''
		from torch_geometric.data import Data
		graph_data = []
		adj_x, adj_y = [], []
		# skip for time step 0
		# lower triangle useful
		total_rows = 0
		for i in range(len(strategies_vec) - 1):
			adj_y.append(np.array(strategies_vec[i+1]))
			num_strategies_in_turn	= int(np.sum(strategies_vec[i]))
			new_matrix		= np.zeros((total_rows + num_strategies_in_turn, total_rows + num_strategies_in_turn))
			new_strategies		= np.zeros((total_rows + num_strategies_in_turn, 1))
			if i != 0:
				new_matrix[: total_rows, : total_rows]	= adj_x[i-1]['matrix']	# copy prev matrix
				new_strategies[: total_rows]		= adj_x[i-1]['strategies']
			curr_row = total_rows
			for stidx, sval in enumerate(strategies_vec[i]):
				if sval == 0: continue
				new_strategies[curr_row, 0]	   = stidx
				#new_strategies.append(stidx)
				new_matrix[curr_row, : total_rows] = 1			# connecting to all in lower half except self
				curr_row += 1
			total_rows = curr_row
			adj_x.append({
				'matrix':	new_matrix,
				'strategies':	new_strategies
			})
			x		= torch.LongTensor(new_strategies)						# (num_strategies, 1) for now. Later will do embedding lookup
			edge_index	= self.get_edge_index_from_adj_matrix(torch.LongTensor(new_matrix))
			y		= torch.FloatTensor(np.array(strategies_vec[i+1]).reshape(1, -1))

			graph_data.append(Data(x = x, edge_index = edge_index, y = y))
		return graph_data

	def get_edge_index_from_adj_matrix(self, adj_matrix):
		from torch_geometric.utils.sparse import dense_to_sparse
		from torch_geometric.utils.undirected import to_undirected
		from torch_geometric.utils.loop import add_self_loops
		edge_index, edge_value = dense_to_sparse(adj_matrix)
		if edge_index.shape[1] != 0 and self.undirected:
			edge_index = to_undirected(edge_index)
		if edge_index.shape[1] != 0 and self.self_loops:
			edge_index, _ = add_self_loops(edge_index)
		return edge_index

if __name__ == "__main__":
	params = {}
	#pdb.set_trace()
	config = {'data_path' : './../../../../data/negotiation_data/data/strategy_vector_data.pkl', 'undirected' : True, 'self_loops': True}
	config = SimpleNamespace(**config)
	data = pickle.load(open(config.data_path, 'rb'))
	data_set = NegotiationGraphDatasetGeometric(data['train'], config)
	sampler = NegotiationDataBatchSampler(data_set.dataset, max_num_utt_batch = 128, shuffle = True, drop_last = False)
	#val_generator = DataLoader(val_set, batch_sampler = sampler, collate_fn = val_set.collate_fn, num_workers = 32)
	data_generator = DataLoader(data_set, batch_sampler = sampler, collate_fn = data_set.collate_fn)

	i = 0
	for epoch in range(5):
		print ('iter ', epoch)
		for data in data_generator:
			# print (data['feats'])
			# print (data['utt_mask'])
			# print (data['ratios'])
			pdb.set_trace()
			#data = data.cuda()
			i += 1
