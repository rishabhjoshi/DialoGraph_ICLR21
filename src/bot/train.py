
# -*- coding: utf-8 -*-
'''
Created on : Monday 25 May, 2020 : 03:40:39
Last Modified : Wednesday 11 Nov, 2020 : 02:43:42

@author       : Rishabh Joshi
Institute     : Carnegie Mellon University
'''
'''
Example Usage: python train.py -data ../../../data/negotiation_data/data/strategy_vector_data_start.pkl -gpu 1 -lr 0.0001 -max_num_utt 64 -model transformer
'''
import warnings
warnings.filterwarnings('ignore')
import faulthandler
faulthandler.enable()
import os
curr_file_path = os.path.dirname(os.path.abspath(__file__)) + '/'
import sys
import pdb
sys.path.append(curr_file_path + '../')
import utils

#from model.simple_rnn import SimpleRNN
#from model.rnn_att import RNNAtt
#from model.transformer import make_model as Transformer
##import sys; sys.path.append('.'); 
from bot_data import NegotiationBotDataset, NegotiationDataBatchSampler
from helper import *
from bot_models import BasicModel
#from torchnlp.metrics import get_moses_multi_bleu
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
#sys.path.insert(0, "/projects/tir1/users/rjoshi2/negotiation/yiheng_negotiation/evaluation/findfeatures/")
sys.path.insert(0, curr_file_path + "yiheng_findfeatures/")
sys.path.insert(0, curr_file_path + '../cocoa_folder/craigslistbargain/')

from dialog_acts_extractor import *
from parse_dialogue import *
from cocoa_folder.cocoa.core.dataset import Example

#from dataset.dialogue_data import NegotiationDataset, NegotiationDataBatchSampler

class Main():

	def load_data(self, file_path):
		# Add code to input data here. This should output train, dev, and test files

		self.data = pickle.load(open(file_path, 'rb'))
		self.p.negotiation_lbl2id = self.data['strategies2colid']
		self.id2strat             = {v:k for k,v in self.p.negotiation_lbl2id.items()}
		self.p.da_lbl2id	= self.data['dialacts2id']
		self.word2id		= self.data['word2id']
		self.id2word		= {v:k for k,v in self.word2id.items()}

		# set feature weights. Not for "start"
		strat_freq = ddict(int)
		da_labels  = []
		tot_labels = 0
		for dat in self.data['train']:
			for strat in dat['strategies_vec']:
				for idx in np.where(np.array(strat) == 1)[0]:
					if idx != self.p.negotiation_lbl2id['<start>']:
						strat_freq[idx] += 1
			for da in dat['dial_acts_vec']:
				if da != self.p.da_lbl2id['<start>']:
					da_labels.append(da)
		tot_labels = len(da_labels)		# these will be without 0
		label_set  = np.unique(da_labels)
		self.strat_feature_weights = {k: ((tot_labels - v) / v) for k,v in strat_freq.items()}
		da_weights		   = compute_class_weight('balanced', label_set, da_labels)
		self.da_feature_weights	   = {label_set[idx]: da_weights[idx] for idx in range(len(label_set))}

		self.p.num_strat	   = len(self.p.negotiation_lbl2id)		- 1	# -1 for start
		self.p.num_da		   = len(self.p.da_lbl2id)			- 1	# -1 for start
		self.p.num_buckets	   = 5

		# Embedding
		self.p.embed = 'no_pretrain'
		self.p.vocab_size = len(self.data['word2id'])
		if self.p.embed == 'pretrain':
			print ("Loading word2vec word embedding...")
			pretrained_word_embedding = list()
			w_model = gensim.models.KeyedVectors.load_word2vec_format("../../../data/glove/GoogleNews-vectors-negative300.bin",binary=True)
			for word in self.data['word2id']:
			        if word not in w_model.wv:
			                pretrained_word_embedding.append(np.random.rand(300,))
			        else:
			                pretrained_word_embedding.append(np.array(w_model.wv[word].tolist()))
			pretrained_word_embedding = np.stack(pretrained_word_embedding, 0)
			self.embedding = pretrained_word_embedding
			#self.embedding = nn.Embedding(len(self.data['word2id']), hidden_size)
			#self.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained_word_embedding).to(device))
			#self.embedding.weight.requires_grad = False
		else:
			self.embedding = np.random.rand(len(self.data['word2id']), 300)

		# for line in tqdm(open('{}/main.json'.format(self.p.data_dir))):
		# 	conv			= json.loads(line)
		# 	_id			= conv['meta']['id']
		# 	num_utter 		= len(conv['transcript'])

		# 	# Get pre-trained embeddings
		# 	conv['embed']	= np.concatenate([pickle.load(open('{}/embeddings/{}/{}.pkl'.format(self.p.data_dir, x, _id), 'rb'))['embeddings'] for x in self.p.embed.split(',')], axis=1)
		# 	conv['labels']	= np.array(conv['labels'])

		# 	self.data[conv['split']].append(conv)
		if self.p.debug:
			self.data['train'] = self.data['train'][:50]

		self.logger.info('\nDataset size -- Train: {}, Valid: {}, Test:{}'.format(len(self.data['train']), len(self.data['valid']), len(self.data['test'])))

		def get_data_loader(split, shuffle=True):
			self.dataset = NegotiationBotDataset(self.data[split], self.p)
			sampler = NegotiationDataBatchSampler(self.dataset.dataset, max_num_utt_batch = self.p.max_num_utt, shuffle = shuffle, drop_last = False)
			return  DataLoader(
					self.dataset,
					batch_sampler   = sampler,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = self.dataset.collate_fn
				)

		self.data_iter = {
			'train'	: get_data_loader('train'),
			'valid'	: get_data_loader('valid', shuffle=False),
			'test'	: get_data_loader('test',  shuffle=False),
		}

	def run_epoch(self, epoch, shuffle = True):
		self.model.train()
		train_losses	= []
		y_pred		= []
		y_true		= []
		all_score	= []
		all_ratio_score	= []
		all_strat_score	= []
		all_da_score	= []
		all_decoded_utt = []
		all_target_utt  = []
		all_ratio_preds = []
		all_ratio_buckets = []
		all_strat_preds = []
		all_strat_tgts  = []
		all_da_preds    = []
		all_da_tgts     = []
		
		#with torch.no_grad():
		bert_scores = {'P':[], 'R':[], 'F1':[]}
		k = 0
		cnt = 0
		for batch in self.data_iter['train']:
			self.optimizer.zero_grad()
			uuids = batch['uuids']
			texts = batch['texts']
			#train_loss, logits, ratio_pred, _		= self.execute(batch)
			#train_loss, decoded_outputs, targets, ratio_preds = self.execute(batch)
			train_loss, decoded_outputs, targets, ratio_preds, strat_out, strat_targets, da_out, da_targets, _ = self.execute(batch)

			decoded_utt = [[self.id2word[idx.item()] for idx in utt] for utt in decoded_outputs]
			decoded_utt = [' '.join(utt).replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip() for utt in decoded_utt]
			decoded_utt = [re.sub('\s+', ' ', utt) for utt in decoded_utt]
			target_utt = [[self.id2word[idx.item()] for idx in utt] for utt in targets]
			target_utt = [' '.join(utt).replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip() for utt in target_utt]
			target_utt = [re.sub('\s+', ' ', utt) for utt in target_utt]
			
			all_decoded_utt += decoded_utt
			all_target_utt  += target_utt
			all_ratio_preds += ratio_preds.cpu().numpy().tolist()
			all_ratio_buckets += batch['ratio_bucket'].view(-1).cpu().numpy().tolist()
			if type(strat_out) == torch.Tensor:
				strat_out = strat_out.detach().cpu().numpy()
				da_out    = da_out.detach().cpu().numpy()
				strat_targets = strat_targets.cpu().numpy()
				da_targets    = da_targets.cpu().numpy()
				all_strat_preds.append(strat_out)
				all_strat_tgts.append(strat_targets)
				all_da_preds.append(da_out)
				all_da_tgts.append(da_targets)

			P,R,F1 = self.bert_scorer.score(decoded_utt, target_utt)
			bert_scores['P'] += P.numpy().tolist()
			bert_scores['R'] += R.numpy().tolist()
			bert_scores['F1'] += F1.numpy().tolist()
			scores = self.get_metrics(decoded_utt, target_utt, ratio_preds.cpu().numpy(), batch['ratio_bucket'].view(-1).cpu().numpy(), strat_out, strat_targets, da_out, da_targets, do_bert_score = False)
			all_score.append(scores[self.p.target])
			all_ratio_score.append(scores['ratio_acc'])
			all_strat_score.append(scores['strat_f1_macro'])
			all_da_score.append(scores['da_acc'])
			#bleu = get_moses_multi_bleu(decoded_utt, target_utt)
			# if decoded_utt[0] == '':
			# 	print(decoded_utt)
			# 	print(decoded_outputs)

			#y_pred_one				= torch.sigmoid(logits[:, 1:, :])

			# if self.p.use_clusters:
			# 	y_pred_one			= torch.softmax(logits[:, :-1, :], dim = 2)
			# 	y_pred_one			= torch.max(y_pred_one, dim = 2)[1]	# 1 is for indices
			# 	y_pred_one			= y_pred_one.unsqueeze(2)		# num_conv x num_utt-1 x 1
			# else:
			# 	y_pred_one				= torch.sigmoid(logits[:, :-1, :])
			# 	y_pred_one[y_pred_one > 0.5]            = 1
			# 	y_pred_one[y_pred_one <= 0.5]           = 0

			train_losses.append(train_loss.item())
			# y_pred.extend(self.get_masked_seq(y_pred_one, 		    batch['utt_mask'][:, 1:]))
			# y_true.extend(self.get_masked_seq(batch['feats'][:, 1:, :], batch['utt_mask'][:, 1:]))
			# all_score.append(self.get_metrics(y_true, y_pred)[self.p.target])

			if (k+1) % self.p.log_freq == 0:
				eval_res = np.round(np.mean(all_score), 4)
				ratio_eval_res = np.round(np.mean(all_ratio_score), 4)
				strat_eval_res = np.round(np.mean(all_strat_score), 4)
				da_eval_res    = np.round(np.mean(all_da_score), 4)
				self.logger.info('[E: {}] | {:.3}% | {} | L: {:.3}, T: BL{:.4} /RB{:.4} /S{:.4} /DA{:.4}, B-V:{:.4}, B-T:{:.4}'.format(epoch, \
					100*cnt/len(self.data['train']),  self.p.name, np.mean(train_losses), eval_res, ratio_eval_res, strat_eval_res, da_eval_res, self.best_val[self.p.target], \
					self.best_test[self.p.target]))
				#print (decoded_utt)

			train_loss.backward()
			self.optimizer.step()

			k	+= 1
			cnt	+= batch['num_conv'][0].item()

		train_loss	= np.mean(train_losses)
		# metrics		= self.get_metrics(y_true, y_pred)
		metrics		= self.get_metrics(all_decoded_utt, all_target_utt, all_ratio_preds, all_ratio_buckets, all_strat_preds, all_strat_tgts, all_da_preds, all_da_tgts, do_bert_score = False, do_roc_auc = True)
		metrics['bert_scores_P'] = np.mean(bert_scores['P'])
		metrics['bert_scores_R'] = np.mean(bert_scores['R'])
		metrics['bert_scores_F1'] = np.mean(bert_scores['F1'])

		return train_loss, metrics

	def execute(self, batch, return_extra = False):
		del batch['uuids']
		del batch['texts']
		batch							= to_gpu(batch, self.device)
		#logitloss, ratioloss, logits, ratio_pred, attn_weights	= self.model(batch, return_extra)
		#decoder_loss, decoded_outputs, targets, ratio_loss, ratio_preds = self.model(batch, return_extra)
		decoder_loss, decoded_outputs, targets, ratio_loss, ratio_preds, strat_loss, strat_out, strat_targets, da_loss, da_out, da_targets, extra = self.model(batch, return_extra)
		if len(self.gpu_list) > 1:
			logitloss = logitloss.mean()
			ratioloss = ratioloss.mean()
		#loss						= self.p.alpha * logitloss + self.p.beta * ratioloss
		loss = decoder_loss + 10*ratio_loss
		if self.p.strat_model != 'none':
			if self.p.fivetimesloss:
				loss += 10 * da_loss + 5 * strat_loss
			else:
				if self.p.no_strat_graph and self.p.no_da_graph:
					pass
				elif self.p.no_strat_graph and not self.p.no_da_graph:
					loss += 10 * da_loss
				elif not self.p.no_strat_graph and self.p.no_da_graph:
					loss += 1 * strat_loss
				else:		# have both strat and da
					loss += 10 * da_loss + 1 * strat_loss

		# ONLY RETURN STRAT EXTRA
		return loss, decoded_outputs, targets, ratio_preds, strat_out, strat_targets, da_out, da_targets, extra#, logits, ratio_pred, attn_weights

	def get_metrics(self, all_decoded_utt, all_target_utt, ratio_preds, ratio_buckets, strat_preds, strat_tgts, da_preds, da_tgts, do_bert_score = True, do_roc_auc = False, return_only_bleus = False):#y_true, y_pred):
		metric = {}
		# P,R,F1 = self.bert_scorer.score(decoded_utt, target_utt)
		# bleu = get_moses_multi_bleu(all_decoded_utt, all_target_utt)
		# if bleu == None:
		# 	bleu = 0
		sm = SmoothingFunction()
		#ref_words = ref_words.split()
		bs = []
		#bertsp, bertsr, bertsf1 = [], [], []
		for i in range(len(all_decoded_utt)):
			ref_words = all_target_utt[i].split()
			ngram_weights = [0.25] * min(4, len(ref_words))
			bleu_score = sentence_bleu([ref_words], all_decoded_utt[i].split(" "), weights=ngram_weights, smoothing_function=sm.method3)
			bs.append(bleu_score)
			# if do_bert_score:
			# 	p, r, f1 = self.bert_scorer.score([all_decoded_utt[i]], [all_target_utt[i]])
			# 	bertsp+=p.numpy().tolist()
			# 	bertsr+=r.numpy().tolist()
			# 	bertsf1+=f1.numpy().tolist()
		if return_only_bleus:
			return bs
		metric['bleu'] = np.mean(bs)
		
		if True:#do_bert_score:
			bertsp, bertsr, bertsf1 = self.bert_scorer.score(all_decoded_utt, all_target_utt)
			metric['bert_scores_P'] = torch.mean(bertsp).item() #np.mean(bertsp.numpy().tolist())
			metric['bert_scores_R'] = torch.mean(bertsr).item() #np.mean(bertsr.numpy().tolist())
			metric['bert_scores_F1'] = torch.mean(bertsf1).item() #np.mean(bertsf1.numpy().tolist())
			# dont put bert scores in full metrics. will put while predicting

		ratio_preds = np.array(ratio_preds)
		ratio_buckets = np.array(ratio_buckets)
		ratio_acc = np.sum(ratio_preds == ratio_buckets) / len(ratio_preds)
		metric['ratio_acc'] = ratio_acc

		if type(strat_preds) == list and len(strat_preds) > 0:
			strat_preds = np.concatenate(strat_preds, 0)
			strat_tgts  = np.concatenate(strat_tgts, 0)
			da_preds    = np.concatenate(da_preds, 0)
			da_tgts     = np.concatenate(da_tgts, 0)

		if type(strat_preds) == np.ndarray and len(strat_preds) > 0:
			strat_preds = sigmoid(strat_preds)
			strat_scores = copy.deepcopy(strat_preds)
			strat_preds[strat_preds >= 0.5] = 1
			strat_preds[strat_preds <  0.5] = 0
			da_preds = softmax(da_preds, axis = 1)
			da_scores = copy.deepcopy(da_preds)
			da_preds = np.argmax(da_preds, axis = 1)
			
			metric['da_acc'] = np.sum(da_preds.flatten() == da_tgts.flatten()) / len(da_preds.flatten())
			metric['strat_report'] = classification_report(strat_tgts, strat_preds, target_names = [x for x,v in self.p.negotiation_lbl2id.items() if x != '<start>'])
			metric['da_report'] = classification_report(da_tgts.flatten(), da_preds.flatten())
			for averaging in ['macro', 'micro', 'weighted']:
				metric['strat_f1_{}'.format(averaging)] = f1_score(strat_tgts, strat_preds, average = averaging)
				metric['da_f1_{}'.format(averaging)] = f1_score(da_tgts.flatten(), da_preds.flatten(), average = averaging)
				if do_roc_auc:
					metric['strat_roc_auc_{}'.format(averaging)] = roc_auc_score(strat_tgts, strat_scores, average = averaging)
					if averaging != 'micro':
						#pass
						metric['da_roc_auc_{}'.format(averaging)] = roc_auc_score(da_tgts.flatten(), da_scores, average = averaging, multi_class = 'ovr')
		else:
			metric['da_acc'] = 0
			metric['strat_report'] = metric['da_report'] = 'NA'
			for averaging in ['macro', 'micro', 'weighted']:
				metric['strat_f1_{}'.format(averaging)] = metric['strat_roc_auc_{}'.format(averaging)] = 0
				metric['da_f1_{}'.format(averaging)] = metric['da_roc_auc_{}'.format(averaging)] = 0
		return metric

	def get_bleu_score(self, decoded_utt, ref_utt):
		sm = SmoothingFunction()
		ref_utt = ref_utt.split()
		ngram_weights = [0.25] * min(4, len(ref_utt))
		bleu = sentence_bleu([ref_words], decoded_utt.split(), weights = ngram_weights, smoothing_function=sm.method3)
		return bleu

	def get_masked_seq(self, seq, mask):
		#masked_seq	= seq.view(-1).masked_select(mask.byte().contiguous().view(-1)).cpu().tolist()
		seq = seq.to(self.device)
		mask = mask.to(self.device)
		mask = (mask.unsqueeze(2) == 1)
		masked_seq = torch.masked_select(seq, mask)
		return masked_seq.cpu().tolist()

	def predict(self, epoch, split, return_extra = False):
		self.model.eval()
		eval_losses, y_pred, y_true	= [], [], []
		all_preds, all_labels, all_masks, all_attention_weights = [], [], [], []
		all_extra_index, all_extra_weight = [], []
		all_extra, all_extra_perm = [], []
		all_extra_gat = []
		all_input_graphs = []
		all_ratio_preds = []
		all_uuids = []
		all_texts = []
		# TODO INCORPORATE SAMPLE WEIGHTS
		all_score = []
		all_ratio_score= []
		all_strat_score = []
		all_da_score    = []
		all_ratio_buckets= []
		all_strat_preds = []
		all_strat_tgts  = []
		all_da_preds    = []
		all_da_tgts     = []
		all_decoded_utt = []
		all_target_utt = []
		bert_scores = {'P':[], 'R':[], 'F1':[]}
		all_metrics = []					# if split is test and return extra is True
		
		with torch.no_grad():
			k = 0
			cnt = 0
			for batch in self.data_iter[split]: # TODO CHECK BATCH AND BATCHES
				self.optimizer.zero_grad()
				self.model.eval()
				uuids = batch['uuids']
				texts = batch['texts']
				#eval_loss, decoded_outputs, targets, ratio_preds = self.execute(batch)
				#eval_loss, decoded_outputs, targets, ratio_preds, strat_out, strat_targets, da_out, da_targets = self.execute(batch)
				eval_loss, decoded_outputs, targets, ratio_preds, strat_out, strat_targets, da_out, da_targets, extra = self.execute(batch, return_extra)

				decoded_utt = [[self.id2word[idx.item()] for idx in utt] for utt in decoded_outputs]
				decoded_utt = [' '.join(utt).replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip() for utt in decoded_utt]
				decoded_utt = [re.sub('\s+', ' ', utt) for utt in decoded_utt]
				target_utt = [[self.id2word[idx.item()] for idx in utt] for utt in targets]
				target_utt = [' '.join(utt).replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip() for utt in target_utt]
				target_utt = [re.sub('\s+', ' ', utt) for utt in target_utt]
				
				all_decoded_utt += decoded_utt
				all_target_utt  += target_utt
				all_ratio_preds += ratio_preds.cpu().numpy().tolist()
				all_ratio_buckets += batch['ratio_bucket'].view(-1).cpu().numpy().tolist()
				if type(strat_out) == torch.Tensor:
					strat_out = strat_out.detach().cpu().numpy()
					da_out    = da_out.detach().cpu().numpy()
					strat_targets = strat_targets.cpu().numpy()
					da_targets    = da_targets.cpu().numpy()
					all_strat_preds.append(strat_out)
					all_strat_tgts.append(strat_targets)
					all_da_preds.append(da_out)
					all_da_tgts.append(da_targets)

				scores = self.get_metrics(decoded_utt, target_utt, ratio_preds.cpu().numpy(), batch['ratio_bucket'].view(-1).cpu().numpy(), strat_out, strat_targets, da_out, da_targets, do_bert_score = False)
				all_score.append(scores[self.p.target])
				all_ratio_score.append(scores['ratio_acc'])
				all_strat_score.append(scores['strat_f1_macro'])
				all_da_score.append(scores['da_acc'])
				
				if extra != None:
					all_extra.append(extra[2].cpu().numpy())	# attn_wts
					all_extra_perm.append(extra[3].cpu().numpy())	# perm
					all_input_graphs.append(extra[4])		# input graph
					if self.p.graph_model == 'gat':
						all_extra_gat.append(extra[5].cpu().numpy())
				all_uuids += uuids
				all_texts += texts

				P,R,F1 = self.bert_scorer.score(decoded_utt, target_utt)
				bert_scores['P'] += P.numpy().tolist()
				bert_scores['R'] += R.numpy().tolist()
				bert_scores['F1'] += F1.numpy().tolist()
				# if return_extra and split == 'test':
				# 	all_metrics.append({'scores': scores, 'bertp': P.numpy().tolist(), 'bertr': R.numpy().tolist(), 'bertf1': F1.numpy().tolist()})

				eval_losses.append(eval_loss.item())
				# y_pred.extend(self.get_masked_seq(y_pred_one, 		    batch['utt_mask'][:, 1:]))
				# y_true.extend(self.get_masked_seq(batch['feats'][:, 1:, :], batch['utt_mask'][:, 1:]))

				# all_preds.append(y_pred_one.cpu().numpy())
				# all_labels.append(batch['feats'][:, 1:, :].cpu().numpy())
				# all_masks.append(batch['utt_mask'][:, 1:].cpu().numpy())
				# all_attention_weights.append(attn_weights.cpu().numpy())

				if (k+1) % self.p.log_freq == 0:
					#eval_res = self.get_metrics(y_true, y_pred)
					#eval_res = {self.p.target : all_score[-1]}
					eval_res = np.round(np.mean(all_score), 4)
					ratio_eval_res = np.round(np.mean(all_ratio_score), 4)
					strat_eval_res = np.round(np.mean(all_strat_score), 4)
					da_eval_res    = np.round(np.mean(all_da_score), 4)
					#print (decoded_utt)
					self.logger.info('[E: {}] | {:.3}% | {} | Eval {} --> L: {:.3}, {}: {:.4}, RB:{:.4}, S:{:.4}, DA:{:.4}'.format(epoch, \
						100*cnt/len(self.data[split]),  self.p.name, split, np.mean(eval_losses), self.p.target, eval_res, ratio_eval_res, strat_eval_res, da_eval_res))
					
				cnt	+= batch['num_conv'][0].item()
				k	+= 1

		#metrics		= self.get_metrics(y_true, y_pred)
		metrics		= self.get_metrics(all_decoded_utt, all_target_utt, all_ratio_preds, all_ratio_buckets, all_strat_preds, all_strat_tgts, all_da_preds, all_da_tgts, do_bert_score = False, do_roc_auc = True)
		metrics['bert_scores_P'] = np.mean(bert_scores['P'])
		metrics['bert_scores_R'] = np.mean(bert_scores['R'])
		metrics['bert_scores_F1'] = np.mean(bert_scores['F1'])
		eval_loss	= np.mean(eval_losses)

		if return_extra and split == 'test':
			bs = self.get_metrics(all_decoded_utt, all_target_utt, all_ratio_preds, all_ratio_buckets, all_strat_preds, all_strat_tgts, all_da_preds, all_da_tgts, do_bert_score = False, do_roc_auc = True, return_only_bleus = True)
			all_metrics = {'all_ratio_preds': all_ratio_preds, 'all_ratio_buckets': all_ratio_buckets, 'all_strat_preds' : all_strat_preds, 'all_strat_tgts': all_strat_tgts, 'all_da_preds': all_da_preds,
					'all_da_tgts': all_da_tgts, 'bert_scores_P': bert_scores['P'], 'bert_scores_R': bert_scores['R'], 'bert_scores_F1': bert_scores['F1'], 'bleus': bs}
			return eval_loss, metrics, all_metrics

		if return_extra:
			return eval_loss, metrics, all_strat_preds, all_strat_tgts, all_extra, all_extra_perm, all_input_graphs, all_extra_gat, all_uuids, all_texts
		return eval_loss, metrics

	def add_optimizer(self, parameters):
		if self.p.opt == 'adam' : return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)
		else                    : return torch.optim.SGD(parameters,  lr=self.p.lr, weight_decay=self.p.l2)

	def add_model(self):
		if	self.p.model.lower() == 'basic':		model = BasicModel(self.p, self.strat_feature_weights, self.da_feature_weights, self.embedding)
		else:   raise NotImplementedError

		model = model.to(self.device)
		if len(self.gpu_list) > 1:
			print ('Using multiple GPUs ', self.p.gpu)
			#model = torch.nn.DataParallel(model, device_ids = list(range(len(self.p.gpu.split(',')))))
			model = torch.nn.DataParallel(model, device_ids = self.p.gpu.split(','))
		return model

	def save_model(self, save_path):
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_test'	: self.best_test,
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}
		torch.save(state, '{}/{}'.format(save_path, self.p.name))		

	def load_model(self, load_path):
		state = torch.load('{}/{}'.format(load_path, self.p.name), map_location = self.device)
		self.best_val		= state['best_val']
		self.best_test		= state['best_test']
		self.best_epoch		= state['best_epoch']

		if len(self.gpu_list) > 1:
			state_dict	= state['state_dict']
			new_state_dict	= OrderedDict()

			for k,v in state_dict.items():
				if 'module' not in k:
					k = 'module.' + k
				else:
					k = k.replace('features.module.', 'module.features.')
				new_state_dict[k] = v

			self.model.load_state_dict(new_state_dict)
		else:
			state_dict	= state['state_dict']
			new_state_dict  = OrderedDict()

			for k,v in state_dict.items():
				if 'module' in k:
					k = k.replace('module.', '')

				new_state_dict[k] = v

			# print(new_state_dict)
			self.model.load_state_dict(new_state_dict)

		if self.p.restore_opt:
			self.optimizer.load_state_dict(state['optimizer'])
		#self.optimizer.load_state_dict(state['optimizer'])
	
	def execute_one(self, raw_example, toks, da_input, strat_input):
		# Create data
		batch = [{'utterance' : toks, 'strategies_vec': strat_input, 'toks_space': toks, 'toks_bert': toks,
			'dial_acts_vec': da_input, 'ratio_bucket': 0, 'uuid': 'lol', 'text': 'lol'}]
		batch = self.dataset.collate_fn(batch)
		del batch['uuids']
		del batch['texts']
		batch = to_gpu(batch, self.device)
		with torch.no_grad():
			decoded_out, strat_out, da_out = self.model(batch, False, only_one = True)
		return decoded_out, strat_out, da_out # strat and da out are the predicted ones

	def chat(self):
		self.model.eval()
		selected_scenario = json.load(open("selected_scenario"))
		price_tracker = PriceTracker('price_tracker.pkl')
		templates = Templates()
		print ("Scenario:")
		print (selected_scenario["kbs"][1]["item"]["Title"] + "\n" + " ".join(selected_scenario["kbs"][1]["item"]["Description"]))
		if self.p.model == 'bert':
			self.tokenizer = BertTokenizer.from_pretrained(self.p.bert_model) # mosly bert based uncased
			toks_space = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[CLS] <start> [SEP]'))]
		else:
			toks_space = [[self.word2id['[CLS]']] + [self.word2id['<start>']] + [self.word2id['[SEP]']]]
		da_input = [0]
		strat_input = np.zeros((1, self.p.num_strat + 1)) # +1 for start
		strat_input[0, -1] = 1.
		raw_example = { "uuid":"C_af8b847888704a0d91b3ad30393c0907", "scenario": selected_scenario,
				"agents":{"1":"human", "0":"human" }, "scenario_uuid":selected_scenario["uuid"],
				"events":[], "outcome": {"reward": 1, "offer": {"price": 13000.0, "sides": ""}} }
		while True:
			try:
				input_sentence = input('> ')
			except KeyError:
				print("Error: Encountered unknown word.")
			if "<accept>" in input_sentence:
				raw_example["events"].append({ "action":"accept", "agent": 0, "time":time.time(), "start_time":time.time() })
				break
			elif "<reject>" in input_sentence: 
				raw_example["events"].append({ "action":"reject", "agent": 0, "time":time.time(), "start_time":time.time() })
				break
			elif "<quit>" in input_sentence:
				raw_example["events"].append({ "action":"quit", "agent": 0, "time":time.time(), "start_time":time.time() })
				break
			elif "<offer>" in input_sentence:
				raw_example["events"].append({ "action":"offer", "agent": 0, "data":{  "price":13000.0, "sides":"" }, "time":time.time(), "start_time":time.time() })
			else:
				raw_example["events"].append({ "action":"message", "agent": 0, "data": input_sentence, "time":time.time(), "start_time":time.time() })

			# Extract DA
			utterance = parse_example(Example.from_dict(raw_example, Scenario), price_tracker, templates)[-1] # -1 for latest
			tmp_dict = utterance.lf.to_dict()
			curr_dial_act = tmp_dict['intent']
			try:
				da_input.append(self.p.da_lbl2id[curr_dial_act])
			except:
				da_input.append(self.p.da_lbl2id['<' + curr_dial_act + '>'])

			_, l_tmp = extract_acts(raw_example)		# extract strategies
			l_tmp = l_tmp[-1] # array of 41
			strat_vec = np.zeros(self.p.num_strat + 1)  # +1 for start
			for sidx, l in enumerate(l_tmp):
				if l == 1:
					try:
						strat_vec[self.p.negotiation_lbl2id[recommendation2uniformstrategymapping[yihengid2recommendation_feature[sidx]]]] = 1
					except:
						continue # who_propose
			strat_input = np.vstack((strat_input, strat_vec))
			print('User Strategies : ', [self.id2strat[sidx] for sidx, s in enumerate(strat_vec) if s == 1])
			input_sentence = utils.normalizeString(input_sentence, tmp_dict, raw_example["scenario"])
			if self.p.model == 'bert':
				toks = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[CLS] '+input_sentence+' [SEP]'))]
			else:
				toks = [self.word2id['[CLS]']] + [self.word2id.get(w, '[UNK]') for w in input_sentence.split()] + [self.word2id['[SEP]']]
			toks_space.append(toks)

			# RUN MODEL
			decoded_out, strat_out, da_out = self.execute_one(raw_example, toks_space, da_input, strat_input) # strat and da out are predicted
			# decoded_words
			#pdb.set_trace()
			if self.p.model == 'bert':
				decoded_words, decoded_out =  index_to_word(decoded_out[-1].cpu().numpy(), raw_example['scenario'], self.tokenizer)
			else:
				decoded_words, decoded_out = index_to_word(decoded_out[-1].cpu().numpy(), raw_example['scenario'], self.id2word)
			# decoded_utt = [[self.id2word[idx.item()] for idx in utt] for utt in decoded_out]
			# decoded_utt = [' '.join(utt).replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip() for utt in decoded_utt]
			# decoded_utt = [re.sub('\s+', ' ', utt) for utt in decoded_utt]

			toks_space.append(decoded_out)
			decoded_words = ' '.join(decoded_words)
			print('Bot:', decoded_words)
			if "<accept>" in decoded_words:
				raw_example["events"].append({ "action":"accept", "agent": 0, "time":time.time(), "start_time":time.time() })
				break
			elif "<reject>" in decoded_words: 
				raw_example["events"].append({ "action":"reject", "agent": 0, "time":time.time(), "start_time":time.time() })
				break
			elif "<quit>" in decoded_words:
				raw_example["events"].append({ "action":"quit", "agent": 0, "time":time.time(), "start_time":time.time() })
				break
			elif "<offer>" in decoded_words:
				raw_example["events"].append({ "action":"offer", "agent": 0, "data":{  "price":13000.0, "sides":"" }, "time":time.time(), "start_time":time.time() })
			else:
				raw_example["events"].append({ "action":"message", "agent": 0, "data": decoded_words, "time":time.time(), "start_time":time.time() })
			# BOT
			utterance = parse_example(Example.from_dict(raw_example, Scenario), price_tracker, templates)[-1] # -1 for latest
			tmp_dict = utterance.lf.to_dict()
			curr_dial_act = tmp_dict['intent']
			try:
				da_input.append(self.p.da_lbl2id[curr_dial_act])
			except:
				da_input.append(self.p.da_lbl2id['<' + curr_dial_act + '>'])

			_, l_tmp = extract_acts(raw_example)		# extract strategies
			l_tmp = l_tmp[-1] # array of 41
			strat_vec = np.zeros(self.p.num_strat + 1)  # +1 for start
			for sidx, l in enumerate(l_tmp):
				if l == 1:
					try:
						strat_vec[self.p.negotiation_lbl2id[recommendation2uniformstrategymapping[yihengid2recommendation_feature[sidx]]]] = 1
					except:
						continue # who_propose
			strat_input = np.vstack((strat_input, strat_vec))
			print('Bot Strategies : ', [self.id2strat[sidx] for sidx, s in enumerate(strat_vec) if s == 1])


	
	def fit(self):
		
		if self.p.restore:
			self.load_model(self.save_dir)
			print ('MODEL LOADED')
			pdb.set_trace()

			# valid_loss, valid_acc, valid_logits, valid_y, valid_trans = self.predict(0, 'valid')
			# res = {
			# 	'data'		: self.data['valid'],
			# 	'transcript'	: mergeList(valid_trans),
			# 	'labels'	: mergeList(valid_y),
			# 	'logits'	: mergeList(valid_logits),
			# 	'acc'		: valid_acc,
			# 	'lbl2id'	: self.lbl2id
			# }
			# pickle.dump(res, open('./visualize/predictions/{}.pkl'.format(self.p.name), 'wb'))
			# exit(0)
			if self.p.only_test:										# For statistical significance
				loss, metrics, all_metrics = self.predict(0, 'test', return_extra = True)
				print ('Score : Loss: {}, {}:{}'.format(loss, self.p.target, metrics[self.p.target]))
				print (metrics)
				dump_dir = './predictions/'; make_dir(dump_dir)
				dump_pickle({
					'loss': loss,
					'metrics': metrics,
					'all_metrics': all_metrics
				}, os.path.join(dump_dir, self.p.name + '_all_metrics_test.pkl'))
				exit(0)

			if self.p.only_eval:										# For graph attention maps
				if self.p.strat_model == 'graph':
					loss, metrics, all_preds, all_labels, all_extra, all_extra_perm, all_input_graphs, all_extra_gat, all_uuids, all_texts = self.predict(0, 'valid', return_extra = True)
				else:
					loss, metrics = self.predict(0, 'valid', return_extra = True)
					all_preds, all_labels, all_extra, all_extra_perm, all_input_graphs, all_extra_gat = None, None, None, None, None, None
				print ('Score: Loss: {}, {}:{}'.format(loss, self.p.target, metrics[self.p.target]))
				print (metrics)

				#dump_dir = os.path.join('./predictions/', self.p.name) + '_predictions.pkl'; make_dir(dump_dir)
				dump_dir = './predictions/'; make_dir(dump_dir)
				dump_pickle({
					'loss': loss,
					'metrics': metrics,
					'all_preds': all_preds,
					'all_labels': all_labels,
					'all_attn_wts': all_extra,
					'all_extra_perm': all_extra_perm,
					'all_input_graphs': all_input_graphs,
					'all_extra_gat': all_extra_gat,
					'all_texts': all_texts,
					'all_uuids': all_uuids
				}, os.path.join(dump_dir, self.p.name + '_predictions.pkl'))

				exit(0)

			if self.p.only_chat:
				while True:
					self.chat()
					try:
						input_sentence = input('Do you want to chat with the bot again?')
					except KeyError:
						print ('Error : Encountered Unknown word')
					if input_sentence.lower() == 'y' or input_sentence.lower() == 'yes':
						continue
					else:
						break
				exit(0)

		kill_cnt		= 0
		for epoch in range(self.p.max_epochs):
			train_loss, train_acc	= self.run_epoch(epoch)
			valid_loss, valid_acc	= self.predict(epoch, 'valid')

			if valid_acc[self.p.target] > self.best_val.get(self.p.target, 0.0):
				self.best_val		= valid_acc
				_, self.best_test	= self.predict(epoch, 'test')
				self.best_epoch		= epoch
				self.save_model(self.save_dir)
				kill_cnt = 0
			else:
				kill_cnt += 1
				self.logger.info('Current kill count : '+str(kill_cnt))
				if kill_cnt > self.p.early_stop:
					self.logger.info('Early Stopping!')
					if self.p.target == 'macro-f1':
						self.logger.info('Best Valid report : \n'+self.best_val['report'])
						self.logger.info('Best Test  report : \n'+self.best_test['report'])
					break

			#if self.p.target == 'macro-f1':
			self.logger.info('Train report : \n'+train_acc['strat_report'])
			self.logger.info('Valid report : \n'+valid_acc['strat_report'])
			self.logger.info('Best Test  report : \n'+self.best_test['strat_report'])

			# self.logger.info('Epoch [{}] | {} | Summary: Train Loss: {:.3}, Train Acc: {}, Valid Acc: {}, Valid Loss: {:.3}, Best valid: {}, Best Test: {}'
			# 		.format(epoch, self.p.name, train_loss, {k:round(v, 4) for k,v in train_acc.items() if k != 'report'}, 
			# 			{k:round(v,4) for k,v in valid_acc.items() if k != 'report'}, valid_loss, 
			# 			{k:round(v,4) for k,v in self.best_val.items() if k != 'report'}, 
			# 			{k:round(v,4) for k,v in self.best_test.items() if k != 'report'}))
			self.logger.info('Epoch [{}] | {} | Summary: Train Loss: {:.3}, Train Acc: {}, Valid Acc: {}, Valid Loss: {:.3}, Best valid: {}, Best Test: {}'
					.format(epoch, self.p.name, train_loss, 
						{k:round(v, 4) for k,v in train_acc.items() if 'report' not in k}, 
						{k:round(v,4) for k,v in valid_acc.items() if 'report' not in k}, 
						valid_loss, 
						{k:round(v,4) for k,v in self.best_val.items() if 'report' not in k}, 
						{k:round(v,4) for k,v in self.best_test.items() if 'report' not in k}))

			self.scheduler.step(valid_loss)
			#self.scheduler.step(valid_acc[self.p.target])

			#self.mongo_log.add_results(self.best_val[self.p.target], self.best_test[self.p.target], self.best_epoch, train_loss)

		self.logger.info('Best Performance: {:.4}'.format(self.best_val[self.p.target]))

	def __init__(self, params):
		self.p = params

		self.save_dir = os.path.join(self.p.model_dir, self.p.log_dir)
		if not os.path.exists(self.p.log_dir): os.system('mkdir -p {}'.format(self.p.log_dir))		# Create log directory if doesn't exist
		if not os.path.exists(self.save_dir):  os.system('mkdir -p {}'.format(self.save_dir))		# Create model directory if doesn't exist

		# Get Logger
		#self.mongo_log	= ResultsMongo(self.p)
		self.logger	= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
		self.logger.info(vars(self.p)); pprint(vars(self.p))

		self.gpu_list = self.p.gpu.split(',')
		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.best_val   = {self.p.target: 0.}
		self.best_test  = {self.p.target: 0., 'strat_report': ''}
		self.best_epoch = 0

		print ('LOADING DATA')
		self.load_data(self.p.dataset)
		print ('LOADING MODEL')
		self.model        = self.add_model()
		self.optimizer    = self.add_optimizer(self.model.parameters())
		#self.scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min')
		self.scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', verbose = True, patience = 2)
		self.bert_scorer  = BERTScorer(lang="en", rescale_with_baseline=True, device = self.device, idf = False)

if __name__== "__main__":

	parser = argparse.ArgumentParser(description='Predicting negotation dialogue strategies')

	parser.add_argument('-data', 	 dest="dataset", 	required=True,							help='Dataset to use')
	parser.add_argument('-gpu', 	 dest="gpu", 		default='0',							help='GPU to use')
	parser.add_argument('-num_workers', dest="num_workers",	default=8,			type=int,			help='Number of dataloader workers')

	parser.add_argument('-model',    dest="model",		default='basic',			 			help='Model to use')
	parser.add_argument('-gru_dim',  dest="gru_hidden_dim", default=300,			type=int, 			help='Hidden state dimension of GRU')
	parser.add_argument('-fc1_weights',dest="fc1_weights",  default=16,			type=int, 			help='Number of hidden states in FC layer for ratio')
	parser.add_argument('-drop',	 dest="dropout", 	default=0.5,  			type=float,			help='Dropout for full connected layer')
	parser.add_argument('-rdrop',	 dest="rec_dropout", 	default=0.5,  			type=float,			help='Recurrent dropout for GRU')
	parser.add_argument('-noweights',dest="noweights", 	action='store_true',						help='To not use feature weights')
	parser.add_argument('-use_clusters',dest="use_clusters",action='store_true',						help='To use clustered embeddings or not. If yes then multi class classification is done.')
	parser.add_argument('-cluster_embed_dim',dest='cluster_embed_dim', default=64,		type=int,			help='Strategy cluster embedding dim')
	parser.add_argument('-num_clusters',dest='num_clusters', default=40,			type=int,			help='Number of clusters to use. Currently 40 and 300')
	parser.add_argument('-debug',	action='store_true',									help='To only use first 50 of train data')
	parser.add_argument('-agent', default='all', help='Which agent to train. Either seller buyer or all')

	parser.add_argument('-utt_enc_hidden',  dest="utt_enc_hidden", default=300,			type=int, 			help='Hidden state dimension of GRU')
	parser.add_argument('-utt_drop',  dest="utt_drop", default=0.3,			type=float, 			help='Dropout after utterance encoder')
	parser.add_argument('-dial_enc_hidden',  dest="dial_enc_hidden", default=300,			type=int, 			help='Hidden state dimension of GRU')
	parser.add_argument('-decoder_hidden',  dest="decoder_hidden", default=300,			type=int, 			help='Hidden state dimension of GRU')
	parser.add_argument('-decoder_drop',  dest="decoder_drop", default=0.1,			type=float, 			help='Dropout after decoder')
	parser.add_argument('-use_bert', action='store_true', help = 'User BERT Encoder or not. Gives some performance boost but slow')
	parser.add_argument('-fix_bert', action='store_true', help = 'Fix BERT encoder. Only applicable if use bert is on')

	parser.add_argument('-strat_model', default='none', type = str, help = 'Strategy Encoder to use')
	parser.add_argument('-strat_hidden', default=300, type=int, help='Strategy hidden size')
	parser.add_argument('-strat_wfst_path', default='../../../data/negotiation_data/data/seq_end_strats_rjyiheng_train_rjyiheng.wfst', type = str, help = 'WFST Path for strategy')
	parser.add_argument('-da_wfst_path', default='../../../data/negotiation_data/data/seq_da_acts_rjyiheng_train_rjyiheng.wfst', type = str, help = 'WFST Path for DA')

	parser.add_argument('-node_feats',dest='node_feats',	default=768,			type=int,			help='Embedding size of nodes')
	parser.add_argument('-ratio',dest='ratio',		default=0.8,			type=float,			help='Ratio in Graph')
	parser.add_argument('-graph_hidden',dest='graph_hidden',default=64,			type=int,			help='Hidden dim in graph')
	parser.add_argument('-graph_layers',dest='graph_layers',default=3,			type=int,			help='Number of Graph Layers')
	parser.add_argument('-graph_drop',dest='graph_drop',	default=0.0,			type=float,			help='Dropout in the Graph Layers')
	parser.add_argument('-num_heads',dest='num_heads',	default=1,			type=int,			help='Number of attention heads in GAT')
	parser.add_argument('-undirected',			action='store_true',						help='Whether to use undirected graph')
	parser.add_argument('-self_loops',			action='store_true',						help='Whether to use self loops')
	parser.add_argument('-node_embed',			action='store_true',						help='Whether to use node embeddings or k hot')
	parser.add_argument('-graph_model',			default='gat',							help='Graph Model')

	parser.add_argument('-fivetimesloss',			action='store_true',						help='Whether to use five times loss or not')
	parser.add_argument('-no_strat_graph',			action='store_true',						help='Ablation study removing strat graph in loss')
	parser.add_argument('-no_da_graph',			action='store_true',						help='Ablation study removing da graph in loss')

	parser.add_argument('-lr',	 dest="lr", 		default=0.001,  		type=float,			help='Learning rate')
	parser.add_argument('-l2', 	 dest="l2", 		default=0.001,  		type=float, 			help='L2 regularization')
	parser.add_argument('-alpha',	 dest="alpha", 		default=1,	 		type=float,			help='Alpha value for logitloss weight')
	parser.add_argument('-beta', 	 dest="beta", 		default=0,			type=float, 			help='Beta value for ratioloss weight')
	parser.add_argument('-epoch', 	 dest="max_epochs", 	default=500,   			type=int, 			help='Max epochs')
	parser.add_argument('-max_num_utt', dest="max_num_utt",	default=64,   			type=int, 			help='Max num of utt in a batch')
	parser.add_argument('-attn',     dest="attn",		default='dot',				 			help='Type of attention')
	parser.add_argument('-restore',	 dest="restore", 	action='store_true', 						help='Restore from the previous best saved model')
	parser.add_argument('-restore_opt',	 dest="restore_opt", 	action='store_true', 					help='Restore optimizer too from the previous best saved model')
	parser.add_argument('-retrain',	 dest="retrain", 	action='store_true', 						help='Retrain from the previous best saved model')
	parser.add_argument('-only_eval',dest="only_eval", 	action='store_true', 						help='Only Evaluate the pretrained model (skip training)')
	parser.add_argument('-only_test',dest="only_test", 	action='store_true', 						help='Only do testing. this is used to obtain all scores for statistical analysis')
	parser.add_argument('-only_chat',dest="only_chat", 	action='store_true', 						help='Only Chat with the model')
	parser.add_argument('-opt',	 dest="opt", 		default='adam', 						help='Optimizer to use for training')
	parser.add_argument('-target',	 dest="target", 	default='strat_f1_macro',						help='Target metric. If use_clusters is set, then this is acc by hardcoding')
	parser.add_argument('-early_stop', dest="early_stop", 	default=5,			type=int,			help='Early stopping')

	parser.add_argument('-log_db',	 default='negotiation',									help='Experiment name')
	parser.add_argument('-eps', 	 dest="eps", 		default=0.00000001,  		type=float, 			help='Value of epsilon')
	parser.add_argument('-name', 	 dest="name", 		default='test_'+str(uuid.uuid4()),				help='Name of the run')
	parser.add_argument('-seed', 	 dest="seed", 		default=1234, 			type=int,			help='Seed for randomization')
	parser.add_argument('-log_freq', dest="log_freq",	default=10, 			type=int,			help='Frequency for logging')
	parser.add_argument('-logdir',	 dest="log_dir", 	default='./log/', 						help='Log directory')
	parser.add_argument('-config',	 dest="config_dir", 	default='./config/', 						help='Config directory')
	parser.add_argument('-modeldir', dest="model_dir", 	default='./save_model/', 					help='Model directory')
	args = parser.parse_args()

	if args.restore: args.name = args.name
	else:		 args.name = args.model + '_' + args.name

	# Set GPU to use
	set_gpu(args.gpu)

	
	# Set seed
	seed_everything(args.seed)
	if args.only_eval or args.only_chat: args.restore = True

	if not args.restore: args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

	# Create Model
	model = Main(args)
	print ('CALLING FIT')
	model.fit()

	if args.retrain:
		del model
		torch.cuda.empty_cache()
		args.restore	 = True
		args.lr		 = 1e-5
		args.restore_opt = True
		
		model = Main(args)
		model.fit()

	print('Model Trained Successfully!!')
