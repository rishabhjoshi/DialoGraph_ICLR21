
# -*- coding: utf-8 -*-
'''
Created on : Monday 25 May, 2020 : 02:27:40
Last Modified : Monday 27 Jul, 2020 : 20:16:21

@author       : Rishabh Joshi
Institute     : Carnegie Mellon University
'''
import sys
sys.path.append('../')
from helper import *
from asap_model import ASAP_Pool
from transformer_model import make_model as Transformer
from higru_model import get_attn_pad_mask
from wfst_model import WFSTModel
from higru_model import UttEncoder
PAD_IDX = 0

class BasicModel(nn.Module):
	def __init__(self, params, strat_feature_weights, da_feature_weights, embedding):
		super(BasicModel, self).__init__()
		num_strats	= len(strat_feature_weights)
		num_da		= len(da_feature_weights)

		# Embedding
		#self.embedding = embedding
		self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
		self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding))
		#self.embedding.weight.requires_grad = False				# Uncomment to fix embed
		self.embed_size = embedding.shape[1]#self.embedding.shape[1] if params.pretrain else 768

		# Utterance Encoder
		self.pretrain = not params.use_bert 
		if not self.pretrain:
			self.bert= BertModel.from_pretrained('bert-base-uncased')#params.bert_model)
			if params.fix_bert:
				for p in self.bert.parameters():
					p.requires_grad = False
			self.bert_remap	      = torch.nn.Linear(in_features = 768, out_features = params.dial_enc_hidden, bias = True)
		else:
			self.utt_encoder = UttEncoder(self.embed_size, params.utt_enc_hidden, 'higru-sf')		# HIGRUSF Encoder

		self.utt_dropout = nn.Dropout(params.utt_drop)
		self.dropout = nn.Dropout(params.dropout)

		self.utt_embed_size = params.utt_enc_hidden # 4 * params.utt_enc_hidden + self.embed_size
		# self.utt_attn	 = Attention(params.utt_enc_hidden, params.attn)
		# self.utt_w	 = torch.nn.Linear(in_features = params.utt_enc_hidden*2, out_features = params.utt_enc_hidden)

		# Dialogue Encoder - Takes a conversation and utt mask and creates dialogue features till that utterance
		# Is a uni GRU cell with Attention for now
		#self.gru_f_cell = torch.nn.GRUCell(self.utt_embed_size, params.dial_enc_hidden)
		self.dial_gru   = torch.nn.GRU(self.utt_embed_size, params.dial_enc_hidden, num_layers = 2, batch_first = True, dropout = 0.3)
		# self.attn	= Attention(params.dial_enc_hidden, params.attn)
		# self.w		= torch.nn.Linear(in_features = params.dial_enc_hidden * 2, out_features = params.dial_enc_hidden)

		# Shapes
		final_dial_hidden_size = params.dial_enc_hidden
		if params.strat_model != 'none':
			final_dial_hidden_size += num_strats + num_da

		# Projection layers
		self.rb_proj_layer    = torch.nn.Linear(in_features = final_dial_hidden_size, out_features = params.num_buckets, bias = True)

		# decoder
		# self.gpt2tokenizer = GTP2Tokenizer.from_pretrained('gpt2')
		# self.decoder_model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id = self.gpt2tokenizer.eos_token_id)
		self.decoder_model = DecoderRNN(self.embedding, final_dial_hidden_size, embedding.shape[0], embedding.shape[1], params.decoder_drop)
		#self.decoder_model = LuongAttnDecoderRNN(params.attn, self.embedding, params.decoder_hidden, embedding.shape[0], 2, params.decoder_drop)

		# Activation
		self.relu = torch.nn.ReLU()

		# Criterions
		if params.noweights:
			self.strat_criterion = torch.nn.BCEWithLogitsLoss(reduction = 'none')
			self.da_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
		else:
			self.strat_criterion = torch.nn.BCEWithLogitsLoss(reduction = 'none', pos_weight = torch.Tensor([x[1] for x in sorted(strat_feature_weights.items(), key = lambda x: x[0])]))
			self.da_criterion = torch.nn.CrossEntropyLoss(reduction = 'none', weight = torch.Tensor([x[1] for x in sorted(da_feature_weights.items(), key = lambda x: x[0])]))
		self.rb_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')

		# Strategy encoder
		self.strat_proj_layer	= torch.nn.Linear(in_features = params.strat_hidden, out_features = params.strat_hidden // 4, bias = True) # -1 for start
		self.da_proj_layer	= torch.nn.Linear(in_features = params.strat_hidden, out_features = params.strat_hidden // 4, bias = True) # -1 for start
		if params.strat_model in ['rnn']:
			self.strat_proj_layer2	= torch.nn.Linear(in_features = params.strat_hidden // 4, out_features = num_strats, bias = True) # -1 for start
			self.da_proj_layer2	= torch.nn.Linear(in_features = params.strat_hidden // 4, out_features = num_da, bias = True) # -1 for start
		else:
			self.strat_proj_layer2	= torch.nn.Linear(in_features = params.strat_hidden, out_features = num_strats, bias = True) # -1 for start
			self.da_proj_layer2	= torch.nn.Linear(in_features = params.strat_hidden, out_features = num_da, bias = True) # -1 for start
		if params.strat_model == 'rnn':
			self.strat_model	= torch.nn.GRU(len(params.negotiation_lbl2id), params.strat_hidden, batch_first = True)
			self.da_model		= torch.nn.GRU(len(params.da_lbl2id), params.strat_hidden, batch_first = True)
		elif params.strat_model == 'fst':
			self.strat_model	= WFSTModel(params.strat_wfst_path, params.negotiation_lbl2id, 'strat')
			self.da_model		= WFSTModel(params.da_wfst_path, params.da_lbl2id, 'da')
		elif params.strat_model == 'transformer':
			self.strat_model	= Transformer(params, tgt_vocab=len(params.negotiation_lbl2id)-1, d_model=len(params.negotiation_lbl2id))
			self.da_model		= Transformer(params, tgt_vocab=len(params.da_lbl2id)-1, d_model=len(params.da_lbl2id))
		elif params.strat_model == 'graph':
			self.strat_model	= ASAP_Pool(params, 'strat')
			self.da_model		= ASAP_Pool(params, 'da')
		elif params.strat_model != 'none':
			raise NotImplementedError

		self.init_weights()
		self.gru_hidden_dim = params.dial_enc_hidden
		self.num_strats	    = num_strats
		self.num_da	    = num_da
		self.which_strat_model    = params.strat_model
		self.agent = params.agent
	
	def init_weights(self):
		torch.nn.init.xavier_uniform_(self.rb_proj_layer.weight)
		torch.nn.init.xavier_uniform_(self.strat_proj_layer.weight)
		torch.nn.init.xavier_uniform_(self.da_proj_layer.weight)
		torch.nn.init.xavier_uniform_(self.strat_proj_layer2.weight)
		torch.nn.init.xavier_uniform_(self.da_proj_layer2.weight)
	
	@property
	def device(self):
		return next(self.parameters()).device

	def forward(self, data, return_extra, only_one = False):
		'''
		data has : input_graph, 
			'ratio_bucket'	: torch.LongTensor(ratio_bucket),		# num_conv x 1
			'num_conv'	: torch.Tensor([num_conv]),			# 1
			'utt_mask'	: torch.FloatTensor(utt_mask),			# num_conv x num_utt
			'strategy_seq'	: torch.FloatTensor(strategy_seq),		# num_conv x num_utt x num_strategies
			'word_input'	: torch.LongTensor(word_input),			# num_conv x num_utt x max_word_seq
			'word_mask'	: torch.FloatTensor(word_mask),			# same
			'bert_input'	: torch.LongTensor(bert_input),			# num_conv x num_utt x max_bert_seq
			'bert_mask'	: torch.FloatTensor(bert_mask),			# same
			'dial_act_input': torch.LongTensor(dial_act_input),		# num_conv x num_utt x 1
			'agent_list'    : torch.LongTensor(agent_list),			# num_conv x num_utt x 1
			'uuids'		: uuids,					# Not torch type
			'texts'		: texts						# Not torch type
		'''
		if only_one:
			return self.forward_one(data)

		num_conv = data['num_conv'][0]
		num_utt  = data['utt_mask'].shape[1]
		
		if self.pretrain:
			w_input		= data['word_input'].view(data['word_input'].shape[0] * data['word_input'].shape[1], data['word_input'].shape[2])
			w_embed		= self.embedding(w_input)			# num_conv * num_utt x num_words x 300
			sa_mask		= get_attn_pad_mask(w_input, w_input)	# num_conv * num_utt x num_words x num_woords
			lens		= torch.sum(w_input != 0, dim = 1)# GET LENS
			utt_outputs	= self.utt_encoder(w_embed, lens, sa_mask)
			utt_outputs	= utt_outputs.view(data['word_input'].shape[0], data['word_input'].shape[1], utt_outputs.shape[-1])
		else:
			w_input		= data['bert_input'].view(data['bert_input'].shape[0] * data['bert_input'].shape[1], data['bert_input'].shape[2])
			mask_input	= data['bert_mask'].view(data['bert_mask'].shape[0] * data['bert_mask'].shape[1], data['bert_mask'].shape[2])
			utt_outputs	= self.bert(input_ids = w_input, attention_mask = mask_input)
			utt_outputs	= utt_outputs[1]
			utt_outputs	= utt_outputs.view(data['bert_input'].shape[0], data['bert_input'].shape[1], utt_outputs.shape[-1])
			utt_outputs	= self.bert_remap(utt_outputs)
		utt_embeds		= self.utt_dropout(utt_outputs)

		#hidden = torch.nn.Parameter(torch.zeros(int(num_conv.item()), self.gru_hidden_dim), requires_grad = True).to(self.device)
		##outs, hs = self.dial_gru(utt_embeds)
		#all_logits = []
		#all_hidden = []
		#for i in range(num_utt):
		#	hidden		= self.gru_f_cell(utt_embeds[:, i, :], hidden)
		#	if i == 0:
		#		logits = self.relu(hidden)
		#		all_logits.append(logits)
		#		all_hidden.append(logits)
		#		continue
		#	attn_weights	= self.attn(hidden.unsqueeze(1), torch.stack(all_hidden, dim = 1))	# num_conv x n_prev
		#	context		= torch.bmm(attn_weights.unsqueeze(1), torch.stack(all_hidden, dim = 1)) # num_conv x 1 x n_prev * num_conv x n_prev x gru_dim ------> num_conv x 1 x gru_dim
		#	cats = self.w(torch.cat((self.relu(hidden), self.relu(context.squeeze(1))), dim = 1)) # num_conv x 2*gru_hidden_dim -> num_conv x gru_hidden_dim
		#	all_logits.append(self.relu(cats))
		#	all_hidden.append(self.relu(hidden))

		#pdb.set_trace()
		dial_encoding, hids = self.dial_gru(utt_embeds)

		lens		= torch.sum(data['utt_mask'], dim = 1)
		target_lens	= (lens - 1).long()				# take 1 off for last - target

		# Strat
		if self.which_strat_model != 'none':
			# Get strategy output
			if self.which_strat_model in ['transformer']:						# Have to pass mask to graph and transformer
				strat_model_out, tempy = self.strat_model(data['strategy_seq'], data['utt_mask'])
			elif self.which_strat_model == 'graph':
				strat_model_out, tempy, strat_extra = self.strat_model(data['strategy_seq'], data['utt_mask'], return_extra)
			else:
				strat_model_out, _ = self.strat_model(data['strategy_seq'])
			da_input		= torch.FloatTensor(data['dial_act_input'].shape[0], data['dial_act_input'].shape[1], self.num_da+1).to(self.device)	# +1 for start input
			da_input.zero_()
			da_input.scatter_(2, data['dial_act_input'], 1)

			# Get da output
			if self.which_strat_model in ['transformer']:
				da_model_out, tempyda = self.da_model(da_input, data['utt_mask'])
			elif self.which_strat_model == 'graph':
				da_model_out, tempyda, da_extra = self.da_model(da_input, data['utt_mask'], return_extra)
			else:
				da_model_out, _ = self.da_model(da_input)

			# take_strat and take_da represent the taken logits (for real utterances | ignoring padding)
			if self.which_strat_model != 'graph':							# Only graph is of the form real_utt x utt_len. Others are num_conv x num_max_utt x utt_len
				take_strat	= strat_model_out[0, :target_lens[0], :]
				take_da		= da_model_out[0, :target_lens[0], :]
			else:
				take_strat	= strat_model_out[:target_lens[0], :]
				take_da		= da_model_out[:target_lens[0], :]
				take_tot = target_lens[0] + 1
			strat_targets	= data['strategy_seq'][0, 1:target_lens[0] + 1, :-1] # remove start at start and start pos
			da_targets	= data['dial_act_input'][0, 1:target_lens[0] + 1, :] # remove start pos
			agent_mask	= data['agent_list'][0, 1:target_lens[0] + 1, :]
			for i in range(1, int(num_conv.item())):
				strat_targets	= torch.cat((strat_targets, data['strategy_seq'][i, 1: target_lens[i] + 1, :-1]), dim = 0)
				da_targets	= torch.cat((da_targets, data['dial_act_input'][i, 1: target_lens[i] + 1, :]), dim = 0)
				agent_mask	= torch.cat((agent_mask, data['agent_list'][i, 1: target_lens[i] + 1, :]), dim = 0)
				if self.which_strat_model != 'graph':
					take_strat	= torch.cat((take_strat, strat_model_out[i, :target_lens[i], :]), dim = 0)
					take_da		= torch.cat((take_da, da_model_out[i, :target_lens[i], :]), dim = 0)
				else:
					take_strat	= torch.cat((take_strat, strat_model_out[take_tot : take_tot + target_lens[i], :]), dim = 0)
					take_da		= torch.cat((take_da, da_model_out[take_tot : take_tot + target_lens[i], :]), dim = 0)
					take_tot	+= target_lens[i] + 1

			if self.which_strat_model == 'rnn':		# only RNN needs projection to num strat and num_da
				strat_out = self.strat_proj_layer2(self.dropout(self.relu(self.strat_proj_layer(self.dropout(take_strat)))))
				da_out    = self.da_proj_layer2(self.dropout(self.relu(self.da_proj_layer(self.dropout(take_da)))))
				#strat_out = self.strat_proj_layer2(self.dropout(take_strat))
				#da_out    = self.da_proj_layer2(self.dropout(take_da))
			else:
				strat_out = take_strat
				da_out    = take_da
			if self.agent == 'buyer':
				agent_mask = 1 - agent_mask
			elif self.agent != 'seller':				# bboth or all
				agent_mask = agent_mask + 1 - agent_mask	# make 1
			strat_loss = self.strat_criterion(strat_out, strat_targets)
			#strat_loss = torch.mean(torch.sum(strat_loss, dim = 1) * agent_mask.view(-1).float())
			strat_loss = torch.mean(torch.sum(strat_loss, dim = 1) )
			da_loss	   = self.da_criterion(da_out, da_targets.squeeze(1))
			#da_loss	   = torch.mean(da_loss * agent_mask.view(-1).float())
			da_loss	   = torch.mean(da_loss )

		# decode  - here all I make num_conv x num_utt x 300 -> num_real_utt x 300 (as all decoded together)
		#dial_encoding	= torch.stack(all_logits, dim = 1)		# num_conv x num_utt x 300
		take_logits	= dial_encoding[0, :target_lens[0], :]
		targets		= data['word_input'][0, 1:target_lens[0] + 1, 1:]	# ignore start add 1 for last turn. Last 1 is for SOS
		agent_mask	= data['agent_list'][0, 1:target_lens[0] + 1, :]
		word_mask	= data['word_mask'][0, 1:target_lens[0] + 1, 1:]
		for i in range(1, int(num_conv.item())):
			take_logits = torch.cat((take_logits, dial_encoding[i, :target_lens[i], :]), dim = 0)
			targets     = torch.cat((targets, data['word_input'][i, 1:target_lens[i]+1, 1:]), dim = 0)
			agent_mask  = torch.cat((agent_mask, data['agent_list'][i, 1: target_lens[i] + 1, :]), dim = 0)
			word_mask   = torch.cat((word_mask, data['word_mask'][i, 1:target_lens[i]+1, 1:]), dim = 0)
		
		if self.which_strat_model != 'none':
			take_logits = torch.cat((take_logits, strat_out, da_out), dim = 1)
		decoded_batch, decoded_outputs = greedy_decode(self.decoder_model, take_logits, None, targets)

		# if decoded_batch[0][0] == 3:
		# 	pdb.set_trace()

		decoder_loss = self.rb_criterion(decoded_outputs.contiguous().view(decoded_outputs.shape[0] * decoded_outputs.shape[1], decoded_outputs.shape[2]), targets.contiguous().view(targets.shape[0] * targets.shape[1]))
		decoder_loss = decoder_loss.view(targets.shape[0], targets.shape[1])
		decoder_loss = decoder_loss * word_mask
		decoded_batch = decoded_batch * word_mask
		# word_mask_len = torch.sum(word_mask, dim = 1) # how many active
		# decoder_loss = torch.sum(decoder_loss, dim = 1) #/ word_mask_len 
		decoder_loss = torch.sum(decoder_loss, dim = 1)			# Sum over an utterance
		if self.agent == 'buyer':
			agent_mask = 1 - agent_mask
		elif self.agent != 'seller':				# bboth or all
			agent_mask = agent_mask + 1 - agent_mask	# make 1
		decoder_loss = decoder_loss * agent_mask.view(-1).float()
		decoder_loss = torch.mean(decoder_loss)				# Mean over num of utterancese
		# TODO Change decoded batch to account for <price>_ and shit for validation


		# RATIO BUCKET
		ratio_logits = dial_encoding[0, target_lens[0], :].view(1, -1)

		if self.which_strat_model != 'none':
			if self.which_strat_model != 'graph':
				take_strat   = strat_model_out[0, target_lens[0], :].view(1, -1)
				take_da	     = da_model_out[0, target_lens[0], :].view(1, -1)
			else:
				take_strat   = strat_model_out[target_lens[0], :].view(1, -1)
				take_da      = da_model_out[target_lens[0], :].view(1, -1)
				take_tot     = target_lens[0] + 1

		for i in range(1, int(num_conv.item())):
			ratio_logits = torch.cat((ratio_logits, dial_encoding[i, target_lens[i], :].view(1, -1)), dim = 0)
			if self.which_strat_model != 'none':
				if self.which_strat_model != 'graph':
					take_strat   = torch.cat((take_strat, strat_model_out[i, target_lens[i], :].view(1, -1)), dim = 0)
					take_da      = torch.cat((take_da, da_model_out[i, target_lens[i], :].view(1, -1)), dim = 0)
				else:
					take_strat   = torch.cat((take_strat, strat_model_out[take_tot + target_lens[i], :].view(1, -1)), dim = 0)
					take_da      = torch.cat((take_da, da_model_out[take_tot + target_lens[i], :].view(1, -1)), dim = 0)
					take_tot     += target_lens[i] + 1
		if self.which_strat_model != 'none':
			if self.which_strat_model == 'rnn':
				strat_out_ratio = self.strat_proj_layer2(self.dropout(self.relu(self.strat_proj_layer(self.dropout(take_strat)))))
				da_out_ratio    = self.da_proj_layer2(self.dropout(self.relu(self.da_proj_layer(self.dropout(take_da)))))
				#strat_out_ratio = self.strat_proj_layer2(self.dropout(take_strat))
				#da_out_ratio    = self.da_proj_layer2(self.dropout(take_da))
			else:
				strat_out_ratio = take_strat
				da_out_ratio    = take_da
			ratio_logits = torch.cat((ratio_logits, strat_out_ratio, da_out_ratio), dim = 1)
		rb_output = self.rb_proj_layer(ratio_logits)
		ratio_loss = self.rb_criterion(rb_output, data['ratio_bucket'].view(-1))
		ratio_loss = torch.mean(ratio_loss)
		ratio_preds = F.softmax(rb_output, 1)
		ratio_preds = torch.argmax(ratio_preds, dim = 1)


		if self.which_strat_model != 'none':
			if return_extra and self.which_strat_model == 'graph':
				return decoder_loss, decoded_batch, targets, ratio_loss, ratio_preds, strat_loss, strat_out, strat_targets, da_loss, da_out, da_targets, strat_extra
			else:
				return decoder_loss, decoded_batch, targets, ratio_loss, ratio_preds, strat_loss, strat_out, strat_targets, da_loss, da_out, da_targets, None
		else:
			return decoder_loss, decoded_batch, targets, ratio_loss, ratio_preds, 0, None, None, 0, None, None, None

	def forward_one(self, data):
		'''
		data has : input_graph, 
			'ratio_bucket'	: torch.LongTensor(ratio_bucket),		# num_conv x 1
			'num_conv'	: torch.Tensor([num_conv]),			# 1
			'utt_mask'	: torch.FloatTensor(utt_mask),			# num_conv x num_utt
			'strategy_seq'	: torch.FloatTensor(strategy_seq),		# num_conv x num_utt x num_strategies
			'word_input'	: torch.LongTensor(word_input),			# num_conv x num_utt x max_word_seq
			'word_mask'	: torch.FloatTensor(word_mask),			# same
			'bert_input'	: torch.LongTensor(bert_input),			# num_conv x num_utt x max_bert_seq
			'bert_mask'	: torch.FloatTensor(bert_mask),			# same
			'dial_act_input': torch.LongTensor(dial_act_input),		# num_conv x num_utt x 1
			'uuids'		: uuids,					# Not torch type
			'texts'		: texts						# Not torch type
		'''
		num_conv = data['num_conv'][0]
		num_utt  = data['utt_mask'].shape[1]
		
		if self.pretrain:
			w_input		= data['word_input'].view(data['word_input'].shape[0] * data['word_input'].shape[1], data['word_input'].shape[2])
			w_embed		= self.embedding(w_input)			# num_conv * num_utt x num_words x 300
			sa_mask		= get_attn_pad_mask(w_input, w_input)	# num_conv * num_utt x num_words x num_woords
			lens		= torch.sum(w_input != 0, dim = 1)# GET LENS
			utt_outputs	= self.utt_encoder(w_embed, lens, sa_mask)
			utt_outputs	= utt_outputs.view(data['word_input'].shape[0], data['word_input'].shape[1], utt_outputs.shape[-1])
		else:
			w_input		= data['bert_input'].view(data['bert_input'].shape[0] * data['bert_input'].shape[1], data['bert_input'].shape[2])
			mask_input	= data['bert_mask'].view(data['bert_mask'].shape[0] * data['bert_mask'].shape[1], data['bert_mask'].shape[2])
			utt_outputs	= self.bert(input_ids = w_input, attention_mask = mask_input)
			utt_outputs	= utt_outputs[1]
			utt_outputs	= utt_outputs.view(data['bert_input'].shape[0], data['bert_input'].shape[1], utt_outputs.shape[-1])
			utt_outputs	= self.bert_remap(utt_outputs)
		utt_embeds		= self.utt_dropout(utt_outputs)

		dial_encoding, hids = self.dial_gru(utt_embeds)

		lens		= torch.sum(data['utt_mask'], dim = 1)
		#target_lens	= (lens - 1).long()				# take 1 off for last - target
		target_lens = lens.long()
		strat_out = None#target_lens = lens.long()
		da_out = None#target_lens = lens.long()

		# Strat
		if self.which_strat_model != 'none':
			# Get strategy output
			if self.which_strat_model in ['transformer']:						# Have to pass mask to graph and transformer
				strat_model_out, tempy = self.strat_model(data['strategy_seq'], data['utt_mask'])
			elif self.which_strat_model == 'graph':
				strat_model_out, tempy, _ = self.strat_model(data['strategy_seq'], data['utt_mask'])
			else:
				strat_model_out, _ = self.strat_model(data['strategy_seq'])
			da_input		= torch.FloatTensor(data['dial_act_input'].shape[0], data['dial_act_input'].shape[1], self.num_da+1).to(self.device)	# +1 for start input
			da_input.zero_()
			da_input.scatter_(2, data['dial_act_input'], 1)

			# Get da output
			if self.which_strat_model in ['transformer']:
				da_model_out, tempyda = self.da_model(da_input, data['utt_mask'])
			elif self.which_strat_model == 'graph':
				da_model_out, tempyda, _ = self.da_model(da_input, data['utt_mask'])
			else:
				da_model_out, _ = self.da_model(da_input)

			# take_strat and take_da represent the taken logits (for real utterances | ignoring padding)
			if self.which_strat_model != 'graph':							# Only graph is of the form real_utt x utt_len. Others are num_conv x num_max_utt x utt_len
				take_strat	= strat_model_out[0, :target_lens[0], :]
				take_da		= da_model_out[0, :target_lens[0], :]
			else:
				take_strat	= strat_model_out[:target_lens[0], :]
				take_da		= da_model_out[:target_lens[0], :]

			if self.which_strat_model == 'rnn':		# only RNN needs projection to num strat and num_da
				strat_out = self.strat_proj_layer2(self.relu(self.strat_proj_layer(take_strat)))
				da_out    = self.da_proj_layer2(self.relu(self.da_proj_layer(take_da)))
				# strat_out = self.strat_proj_layer2(self.dropout(take_strat))
				# da_out    = self.da_proj_layer2(self.dropout(take_da))
			else:
				strat_out = take_strat
				da_out    = take_da

		# decode  - here all I make num_conv x num_utt x 300 -> num_real_utt x 300 (as all decoded together)
		take_logits	= dial_encoding[0, :target_lens[0], :]
		targets		= None
		if self.which_strat_model != 'none':
			take_logits = torch.cat((take_logits, strat_out, da_out), dim = 1)
		decoded_batch, decoded_outputs = greedy_decode(self.decoder_model, take_logits, None, targets, evaluate = True)

		return decoded_batch, strat_out, da_out


class Attention(nn.Module):
	''' based on global attention model by 1508.04025 '''
	def __init__(self, hidden_dim, method):
		super(Attention, self).__init__()
		self.method = method
		self.hidden_dim = hidden_dim

		if method == 'general':
			self.w = nn.Linear(hidden_dim, hidden_dim)
		elif method == 'concat':
			self.w = nn.Linear(hidden_dim*2, hidden_dim)
			self.v = torch.nn.Parameter(torch.FloatTensor(hidden_dim))

	def forward(self, dec_out, enc_outs): # num_conv x 1 x gru_dim and num_conv x k x gru_dim
		if self.method == 'dot':
			attn_energies = self.dot(dec_out, enc_outs)
		elif self.method == 'general':
			attn_energies = self.general(dec_out, enc_outs)
		elif self.method == 'concat':
			attn_energies = self.concat(dec_out, enc_outs)
		#return F.softmax(attn_energies, dim=0)
		return F.softmax(attn_energies, dim=1)				# CHANGED THIS # numconv x k
	
	def dot(self, dec_out, enc_outs):
		return torch.sum(dec_out*enc_outs, dim=2)			# num_conv x k

	def general(self, dec_out, enc_outs):
		energy = self.w(enc_outs)
		return torch.sum(dec_out*energy, dim=2)

	def concat(self, dec_out, enc_outs):
		dec_out = dec_out.expand(enc_outs.shape[0], -1, -1)
		energy = torch.cat((dec_out, enc_outs), 2)
		return torch.sum(self.v * self.w(energy).tanh(), dim=2)


SOS_token = 2
EOS_token = 3

class DecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, embedding_size, dropout=0.1):
        '''
        Illustrative decoder
        '''
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding 

        self.rnn = nn.GRU(embedding_size, hidden_size, bidirectional=False, dropout=dropout, batch_first=False)
        self.dropout_rate = dropout
        self.out = nn.Linear(hidden_size*1, output_size)		# *2 for bidirectional

    def forward(self, input, hidden, not_used):
        embedded = self.embedding(input).transpose(0, 1)  # [B,1] -> [ 1, B, D]
        embedded = F.dropout(embedded, self.dropout_rate)

        output = embedded

        output, hidden = self.rnn(output, hidden)

        out = self.out(output.squeeze(0))
        output = F.log_softmax(out, dim=1)
        return output, hidden

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

def beam_decode(target_tensor, decoder_hiddens, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 10
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[SOS_token]], device=device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch


def greedy_decode(decoder, decoder_hidden, encoder_outputs, target_tensor, evaluate = False):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''
    device = decoder_hidden.device
    if evaluate == True:
        batch_size = decoder_hidden.shape[0]
        seq_len = 100
    else:
        batch_size, seq_len = target_tensor.size()
    MAX_LENGTH = seq_len
    decoded_batch = torch.zeros((batch_size, MAX_LENGTH)).to(device)
    decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)]).to(device)
    decoder_outputs = []

    #decoder_hidden = torch.cat((decoder_hidden.unsqueeze(0), decoder_hidden.unsqueeze(0)), dim=0) # 2 x bs x 300 (2 for bidirectional)
    decoder_hidden = decoder_hidden.unsqueeze(0)

    for t in range(MAX_LENGTH):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_outputs.append(decoder_output)

        topv, topi = decoder_output.data.topk(1)  # get candidates
        topi = topi.view(-1)
        decoded_batch[:, t] = topi
        #pdb.set_trace()

        if evaluate: # NO TEACHER FORCING
            decoder_input = topi.detach().view(-1, 1)
        else:
            decoder_input = target_tensor[:, t].view(-1, 1)

    return decoded_batch, torch.stack(decoder_outputs, 0).permute(1, 0, 2)

