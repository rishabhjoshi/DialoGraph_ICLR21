
# -*- coding: utf-8 -*-
'''
Created on : Monday 29 Jun, 2020 : 02:07:57
Last Modified : Monday 29 Jun, 2020 : 08:14:45

@author       : Rishabh Joshi
Institute     : Carnegie Mellon University
'''
# -*- coding: utf-8 -*-
import faulthandler
faulthandler.enable()
import random
import re, pdb
from collections import defaultdict
from src.basic.sample_utils import sample_candidates
from session import Session
from src.model.preprocess import tokenize, word_to_num
from src.model.vocab import is_entity
# from src.basic.lexicon import Lexicon
import numpy as np
import csv, os
from subprocess import Popen, PIPE
import subprocess
#from transformers import BertTokenizer

import sys
import copy
curr_file_path = os.path.dirname(os.path.abspath(__file__)) + '/'
sys.path.append(str(curr_file_path + '../../../../../'))
sys.path.append(str(curr_file_path + '../../../../'))
import utils
from helper import *
from bot_models import BasicModel
sys.path.insert(0, str(curr_file_path + '../../../../yiheng_findfeatures/'))
sys.path.insert(0, str(curr_file_path + '../../../../../cocoa_folder/craigslistbargain/'))
from dialog_acts_extractor import *
from parse_dialogue import *
from cocoa_folder.cocoa.core.dataset import Example
from types import SimpleNamespace
curr_file_path = os.path.dirname(os.path.abspath(__file__)) + '/'           # again because overrridden in one file


class SimpleSession(Session):
    '''
    The simple system implements a bot that acts as the seller
    '''
    def __init__(self, agent, kb, lexicon, style, realizer=None, consecutive_entity=True, gpu = 0, strat_model = 'graph'):
        super(SimpleSession, self).__init__(agent)
        self.agent = agent
        ##cmd = "PYTHONPATH='/usr1/home/rjoshi2/negotiation_personality/src/negotiation/bot' python /usr1/home/rjoshi2/negotiation_personality/src/negotiation/bot/train.py -data /usr1/home/rjoshi2/negotiation_personality/data/negotiation_data/data/strategy_vector_data_FULL_Yiheng.pkl -gpu -1 -lr 0.001 -max_num_utt 128 -model basic -num_workers 0 -strat_model none -agent seller -use_bert -fix_bert -name basic_final_none_seller_BERT_28_06_2020_05:48:40 -only_chat -restore"
        #my_env = os.environ.copy()
        #my_env['PYTHONPATH'] = '/usr1/home/rjoshi2/negotiation_personality/src/negotiation/bot'
        #cmd = "python /usr1/home/rjoshi2/negotiation_personality/src/negotiation/bot/train.py -data /usr1/home/rjoshi2/negotiation_personality/data/negotiation_data/data/strategy_vector_data_FULL_Yiheng.pkl -gpu -1 -lr 0.001 -max_num_utt 128 -model basic -num_workers 0 -strat_model none -agent seller -use_bert -fix_bert -name basic_final_none_seller_BERT_28_06_2020_05:48:40 -only_chat -restore"
        ##self.process = Popen(cmd.split(), stdin=PIPE, stdout=PIPE, env=my_env)
        ##self.process = Popen(cmd.split(),  env=my_env)
        #self.process = subprocess.call(cmd, shell = True)
        self.bot = strat_model
        params = {'model': 'bert', 'gru_dim': 300, 'fc1_weights': 16, 'dropout': 0.5, 'noweights': False, 'use_clusters': False, 'agent': 'seller', 'rec_dropout': 0.5,
                'utt_enc_hidden': 300, 'utt_drop': 0.3, 'dial_enc_hidden': 300, 'decoder_hidden': 300, 'decoder_drop': 0.1, 'data': curr_file_path + '../../../../../../../data/negotiation_data/data/strategy_vector_data_FULL_Yiheng.pkl',
                'use_bert': True, 'fix_bert': True, 'strat_hidden': 300, 'attn': 'dot', 
                'seed': 1234, 'log_dir': './log/', 'model_dir': curr_file_path + '../../../../save_model'}
        if self.bot == 'hed':
            params['strat_model'] = 'none'
            params['name'] = 'basic_final_none_seller_BERT_28_06_2020_05:48:40'
            print ('Loading HED Model')
        elif self.bot == 'fst':
            params['strat_model'] = 'fst'
            params['strat_wfst_path'] = curr_file_path + '../../../../../../../data/negotiation_data/data/seq_end_strats_rjyiheng_train_rjyiheng.wfst'
            params['da_wfst_path'] = curr_file_path + '../../../../../../../data/negotiation_data/data/seq_da_acts_rjyiheng_train_rjyiheng.wfst'
            params['name'] = 'basic_final_fst_seller_BERT_28_06_2020_06:01:00'
            print ('Loading FST Model')
        elif self.bot == 'rnn':
            params['strat_model'] = 'rnn'
            params['strat_hidden'] = 64
            params['name'] = 'basic_final_rnn_seller_BERT_64_30_06_2020_01:00:15'
            print ('Loading RNN Model')
        elif self.bot == 'transformer':
            params['strat_model'] = 'transformer'
            params['name'] = 'basic_final_transformer_seller_BERT_28_06_2020_06:01:00'
            print ('Loading Transformer Model')
        elif self.bot == 'graph':
            params['strat_model'] = 'graph'
            params['ratio'] = 0.8
            params['graph_hidden'] = 64
            params['graph_drop'] = 0.0
            params['graph_layers'] = 2
            params['num_heads'] = 1
            params['self_loops'] = True
            params['node_embed'] = False
            params['undirected'] = False
            params['graph_model'] = 'gat'
            params['name'] = 'basic_final_graph_seller_gat_64_08_2_001_BERT_28_06_2020_06:01:24'
            print ('Loading Graph Model')
        else:
            raise NotImplementedError
        self.p = SimpleNamespace(**params)
        # data
        self.data = load_pickle(self.p.data)
        self.p.negotiation_lbl2id = self.data['strategies2colid']
        self.id2strat = {v:k for k,v in self.p.negotiation_lbl2id.items()}
        self.p.da_lbl2id    = self.data['dialacts2id']
        self.word2id        = self.data['word2id']
        self.id2word        = {v:k for k,v in self.word2id.items()}

        self.p.num_strat       = len(self.p.negotiation_lbl2id)        - 1    # -1 for start
        self.p.num_da          = len(self.p.da_lbl2id)          - 1 # -1 for start
        self.p.num_buckets       = 5

        #model
        self.save_dir = os.path.join(self.p.model_dir, self.p.log_dir)
        if gpu != -1 and gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')
        self.model = self.add_model()
        self.restore_model()
        self.model.eval()
        self.price_tracker = PriceTracker(curr_file_path + '../../../../price_tracker.pkl')
        self.templates = Templates()

        # usables
        if self.p.model == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.toks_space = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[CLS] <start> [SEP]'))]
        else:
            self.toks_space = [[self.word2id['[CLS]']] + [self.word2id['<start>']] + [self.word2id['[SEP]']]]
        self.da_input = [0]
        self.strat_input = np.zeros((1, self.p.num_strat + 1)) # +1 for start
        self.strat_input[0, -1] = 1.
        self.agent_list = [-1]
        selected_scenario = json.load(open(curr_file_path + "../../../../selected_scenario"))

        self.raw_example = { "uuid":"C_af8b847888704a0d91b3ad30393c0907", "scenario": selected_scenario,
                    "agents":{"1":"human", "0":"human" }, "scenario_uuid":selected_scenario["uuid"],
                    "events":[], "outcome": {"reward": 1, "offer": {"price": 13000.0, "sides": ""}} }
        self.first_chance = True
        self.to_send = True
        self.end = False
        self.decoded_words = ''

    def send(self):
        if self.to_send:
            self.to_send = False
            if self.first_chance:
                self.first_chance = False
                #raise NotImplementedError                           # THIS IS CALLED LAST
                if self.bot == 'hed':
                    return self.message('Hello.')
                elif self.bot == 'fst':
                    return self.message('Hello..')
                elif self.bot == 'rnn':
                    return self.message('Hello!')
                elif self.bot == 'transformer':
                    return self.message('Hello!!')
                else :
                    return self.message('Hello!!!')
            else:
                #raise NotImplementedError
                return self.message(self.decoded_words)
        else:
            self.to_send = True
            return None

    def receive(self, event):
        if self.end == True:
            pass
        if event.action == 'message':
            input_sentence = event.data
            print ('Utterance received : ', input_sentence)
            if "<accept>" in input_sentence:
                self.raw_example["events"].append({ "action":"accept", "agent": 0, "time":time.time(), "start_time":time.time() , "data": None})
                self.end = True
                return
            elif "<reject>" in input_sentence: 
                self.raw_example["events"].append({ "action":"reject", "agent": 0, "time":time.time(), "start_time":time.time() , "data": None})
                self.end = True
                return
            elif "<quit>" in input_sentence:
                self.raw_example["events"].append({ "action":"quit", "agent": 0, "time":time.time(), "start_time":time.time() , "data": None})
                self.end = True
                return
            elif "<offer>" in input_sentence:
                self.raw_example["events"].append({ "action":"offer", "agent": 0, "data":{  "price":13000.0, "sides":"" }, "time":time.time(), "start_time":time.time() })
            else:
                self.raw_example["events"].append({ "action":"message", "agent": 0, "data": input_sentence, "time":time.time(), "start_time":time.time() })

            # Extract DA
            utterance = parse_example(Example.from_dict(self.raw_example, Scenario), self.price_tracker, self.templates)[-1] # -1 for latest
            tmp_dict = utterance.lf.to_dict()
            curr_dial_act = tmp_dict['intent']
            try:
                self.da_input.append(self.p.da_lbl2id[curr_dial_act])
            except:
                self.da_input.append(self.p.da_lbl2id['<' + curr_dial_act + '>'])

            _, l_tmp = extract_acts(self.raw_example)
            l_tmp = l_tmp[-1] # array of 41
            strat_vec = np.zeros(self.p.num_strat + 1) # +1 for start
            for sidx, l in enumerate(l_tmp):
                if l == 1:
                    try:
                        strat_vec[self.p.negotiation_lbl2id[recommendation2uniformstrategymapping[yihengid2recommendation_feature[sidx]]]] = 1
                    except:
                        continue # who_propose
            self.strat_input = np.vstack((self.strat_input, strat_vec))

            input_sentence = utils.normalizeString(input_sentence, tmp_dict, self.raw_example["scenario"])
            if self.p.model == 'bert':
                toks = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[CLS] '+input_sentence+' [SEP]'))
            else:
                toks = [self.word2id['[CLS]']] + [self.word2id.get(w, '[UNK]') for w in input_sentence.split()] + [self.word2id['[SEP]']]
            self.toks_space.append(toks)
            self.agent_list.append(0)

            # RUN MODEL
            decoded_out, strat_out, da_out = self.execute_one() # strat and da out are predicted
            decoded_words, decoded_out = index_to_word(decoded_out[-1].cpu().numpy(), self.raw_example['scenario'], self.id2word, model = 'basic')

            self.toks_space.append(decoded_out)
            self.agent_list.append(1)
            decoded_words = ' '.join(decoded_words)
            self.decoded_words = decoded_words
            if "<accept>" in decoded_words:
                self.raw_example["events"].append({ "action":"accept", "agent": 0, "time":time.time(), "start_time":time.time() , "data": None})
                self.end = True
                return
            elif "<reject>" in decoded_words: 
                self.raw_example["events"].append({ "action":"reject", "agent": 0, "time":time.time(), "start_time":time.time() , "data": None})
                self.end = True
                return
            elif "<quit>" in decoded_words:
                self.raw_example["events"].append({ "action":"quit", "agent": 0, "time":time.time(), "start_time":time.time() , "data": None})
                self.end = True
                return
            elif "<offer>" in decoded_words:
                self.raw_example["events"].append({ "action":"offer", "agent": 0, "data":{  "price":13000.0, "sides":"" }, "time":time.time(), "start_time":time.time() })
            else:
                self.raw_example["events"].append({ "action":"message", "agent": 0, "data": decoded_words, "time":time.time(), "start_time":time.time() })

            # BOT
            utterance = parse_example(Example.from_dict(self.raw_example, Scenario), self.price_tracker, self.templates)[-1] # -1 for latest
            tmp_dict = utterance.lf.to_dict()
            curr_dial_act = tmp_dict['intent']
            try:
                self.da_input.append(self.p.da_lbl2id[curr_dial_act])
            except:
                self.da_input.append(self.p.da_lbl2id['<' + curr_dial_act + '>'])

            _, l_tmp = extract_acts(self.raw_example)        # extract strategies
            l_tmp = l_tmp[-1] # array of 41
            strat_vec = np.zeros(self.p.num_strat + 1)  # +1 for start
            for sidx, l in enumerate(l_tmp):
                if l == 1:
                    try:
                        strat_vec[self.p.negotiation_lbl2id[recommendation2uniformstrategymapping[yihengid2recommendation_feature[sidx]]]] = 1
                    except:
                        continue # who_propose
            self.strat_input = np.vstack((self.strat_input, strat_vec))

        elif event.action == 'select':
            raise NotImplementedError

    def add_model(self):
        # The feature weights are not used. Can put anythign similar to the training ones. 
        strat_feature_weights = {1: 9.989507299270073, 2: 1.8890621252098825, 3: 3.0165068778657775, 10: 0.8896297458424851, 12: 6.981776010603048, 19: 0.9693427076520602, 20: 4.781591263650546, 0: 1.0151413752718754, 7: 4.4071829405162735, 9: 18.379726468222042, 13: 3.7275046609753706, 16: 1.650346572780284, 21: 15.952146375791695, 14: 13.296142433234422, 11: 27.49083382613838, 4: 12.316196793808734, 6: 19.29401853411963, 15: 14.270364500792393, 18: 133.95238095238096, 5: 53.935005701254276, 17: 243.55837563451777, 8: 328.986301369863}
        da_feature_weights = {0: 1.428512127142264, 1: 1.7397804420049112, 2: 0.7851439001336332, 3: 13.93233082706767, 4: 0.4299994644865318, 5: 8.603214285714285, 6: 0.6678217959025256, 7: 0.3579452584029243, 8: 7.2908595641646485, 9: 0.8592473693597289, 10: 0.7752389534322403, 11: 1.6770398217766638, 12: 0.7323442677773386, 13: 10.621252204585538}
        embedding = np.random.rand(len(self.data['word2id']), 300)
        model = BasicModel(self.p, strat_feature_weights, da_feature_weights, embedding)
        model = model.to(self.device)
        return model

    def restore_model(self):
        load_path = self.save_dir
        state = torch.load('{}/{}'.format(load_path, self.p.name), map_location = self.device)
        state_dict = state['state_dict']
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            if 'module' in k:
                k = k.replace('module.', '')
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)

    def execute_one(self):
        #takes self.raw_example, self.toks_space, self.da_input, self.strat_input
        batch = [{'utterance' : self.toks_space, 'strategies_vec': self.strat_input, 'toks_space': self.toks_space, 'toks_bert': self.toks_space,
            'dial_acts_vec': self.da_input, 'ratio_bucket': 0, 'uuid': 'lol', 'text': 'lol', 'agent_list': self.agent_list}] 
        try:
            batch = self.collate_fn(batch)
        except:
            toks = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[CLS] '+'ok'+' [SEP]'))
            print ('Error Here', batch, self.raw_example)
            return toks, None, None
        del batch['uuids']
        del batch['texts']
        batch = to_gpu(batch, self.device)
        with torch.no_grad():
            decoded_out, strat_out, da_out = self.model(batch, False, only_one = True)
        return decoded_out, strat_out, da_out
    
    def collate_fn(self, batch):
        # [ {}, {}]
        # [ (strategies_vec[0], ratio[0]), (strategies_vec[1], ratio[1]), ... ()]
        num_conv = len(batch)
        num_utt = np.max([len(b['utterance']) for b in batch])                # max num utt in any conv
        num_strategies = batch[0]['strategies_vec'].shape[1]                # 1 for clustered strategies
        max_word_seq = np.max([len(x) for b in batch for x in b['toks_space']])
        max_bert_seq = np.max([len(x) for b in batch for x in b['toks_bert']])
        #max_word_seq = np.max([len(b['toks_space']) for b in batch])
        #max_bert_seq = np.max([len(b['toks_bert']) for b in batch])

        strategy_seq    = np.zeros((num_conv, num_utt, num_strategies))
        utt_mask    = np.zeros((num_conv, num_utt))
        ratio_bucket    = np.zeros((num_conv, 1))
        word_input    = np.zeros((num_conv, num_utt, max_word_seq))
        word_mask    = np.zeros((num_conv, num_utt, max_word_seq))
        bert_input    = np.zeros((num_conv, num_utt, max_bert_seq))
        bert_mask    = np.zeros((num_conv, num_utt, max_bert_seq))
        da_input    = np.zeros((num_conv, num_utt, 1))
        agent_list    = np.full((num_conv, num_utt, 1), -1)
        uuids        = [b['uuid'] for b in batch]
        texts        = [b['utterance'] for b in batch]

        for i in range(num_conv):
            curr_utt_num        = len(batch[i]['utterance'])
            for j in range(curr_utt_num):
                curr_toks_num = max(len(batch[i]['toks_space'][j]), len(batch[i]['toks_bert'][j]))#.shape[1]
                word_input[i, j, 0:curr_toks_num]        = batch[i]['toks_space'][j]
                word_mask[i, j, 0:curr_toks_num]        = 1.0
                bert_input[i, j, 0:curr_toks_num]        = batch[i]['toks_bert'][j]
                bert_mask[i, j, 0:curr_toks_num]        = 1.0

            strategy_seq[i, 0:curr_utt_num, :]                = batch[i]['strategies_vec']
            utt_mask[i, 0:curr_utt_num]                    = 1.0
            ratio_bucket[i][0]                        = batch[i]['ratio_bucket']
            da_input[i, 0:curr_utt_num, 0]                    = batch[i]['dial_acts_vec']
            agent_list[i, 0:curr_utt_num, 0]                = batch[i]['agent_list']

        batch = {
            #'feats'        : torch.FloatTensor(input_batch), 
            'ratio_bucket'    : torch.LongTensor(ratio_bucket), 
            'num_conv'    : torch.Tensor([num_conv]),
            'utt_mask'    : torch.FloatTensor(utt_mask),
            'strategy_seq'    : torch.FloatTensor(strategy_seq),
            'word_input'    : torch.LongTensor(word_input),
            'word_mask'    : torch.FloatTensor(word_mask),
            'bert_input'    : torch.LongTensor(bert_input),
            'bert_mask'    : torch.FloatTensor(bert_mask),
            'dial_act_input': torch.LongTensor(da_input),
            'agent_list'    : torch.LongTensor(agent_list),
            'uuids'        : uuids,                    # Not torch type
            'texts'        : texts                        # Not torch type
        }
        #return torch.FloatTensor(input_batch), torch.LongTensor(utt_mask), torch.FloatTensor(ratios)
        return batch#, uuids, texts
