
# -*- coding: utf-8 -*-
'''
Created on : Friday 19 Jun, 2020 : 18:46:56
Last Modified : Friday 19 Jun, 2020 : 18:48:07

@author       : Rishabh Joshi
Institute     : Carnegie Mellon University
'''
import sys
sys.path.append('../')
from helper import *
#### WFST ###
class WFSTModel(torch.nn.Module):
    '''
    This module takes feature vectors
    Runs a WFST over them
    Encoder
    '''
    def __init__(self, fst_path, strategy2idx, strat_or_da):
        super().__init__()
        self.wfsm = wfst(fst_path)
        self.strategy2idx = strategy2idx
        self.idx2strategy = {v:k for k,v in strategy2idx.items()}
        self.projection_layer = torch.nn.Linear(in_features=len(strategy2idx), out_features=len(strategy2idx)-1,bias=True)
        self.relu = torch.nn.ReLU()
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)
        self.strat_or_da = strat_or_da

    def forward(self, feats, teacher_forcing = True, last = True):# utt_mask, ratios):
        '''
        feats : num_conv x num_utt x num_feats		# num_feats is 1 for clustered data
        '''
        #feats = feats.unsqueeze(0)
        num_conv = feats.shape[0]
        num_utt  = feats.shape[1]
        num_feats = feats.shape[2]
        #pdb.set_trace()
        preds = torch.FloatTensor(np.zeros((num_conv, num_utt, num_feats-1))).to(feats.device)
        for conv_id in range(num_conv):
            self.wfsm.current_state = 0		# reset WFST
            # for first conv, dont see preds, just give input
            #pdb.set_trace()
            for f_id, f in enumerate(feats[conv_id][0]):
                if f_id == 0:
                    self.wfsm.step('<<start>>')
                if f == 1:
                    self.wfsm.step('<'+self.idx2strategy[f_id]+'>')
            if self.strat_or_da == 'strat':
                self.wfsm.step('<end>')
            # for all other conv, see expected output and match with input features
            for utt_id in range(1, num_utt):
                # give agent id true anyway
                preds[conv_id][utt_id-1][0] = feats[conv_id][utt_id][0]
                # see predicted strategies
                if self.strat_or_da == 'strat':
                    predicted_states, predicted_strategies, predicted_embeddings = self.wfsm.get_seq_till_next_utt()
                    predicted_embeddings = np.concatenate(predicted_embeddings, 0)
                    predicted_embeddings = np.max(predicted_embeddings, 0)
                    predicted_embeddings = torch.FloatTensor(predicted_embeddings).to(feats.device).view(1, -1)
                    preds[conv_id][utt_id-1] = self.projection_layer(predicted_embeddings)
                else:
                    curr_state = self.wfsm.current_state
                    e = self.wfsm.look_up_state_embedding(curr_state)
                    e = torch.FloatTensor(e).to(feats.device).view(1, -1)
                    preds[conv_id][utt_id-1] = self.projection_layer(e)
                # strategy_idx = np.random.choice(np.arange(len(e)), p=e / np.sum(e))
                # strategy = self.wfsm.get_strategy_from_index(strategy_idx)
                # preds[conv_id][utt_id-1][self.strategy2idx[strategy]] = 1
                # # teacher force the true strategies
                if teacher_forcing:
                    for f_id, f in enumerate(feats[conv_id][utt_id]):
                        if f_id == 0: continue
                        if f == 1:
                            self.wfsm.step('<'+self.idx2strategy[f_id]+'>')
                elif self.strat_or_da == 'da':
                    self.wfsm.step('<'+strategy+'>')
                else:
                    for pred_strat in predicted_strategies:
                        self.wfsm.step('<'+pred_strat+'>')
                    self.wfsm.step('<end>')
        #logits = torch.FloatTensor(preds[:, -1, :].reshape(1, -1)).to(feats.device)
        return preds, None

#Class for tracking current states for FST
class wfst:
    def __init__(self, filename="/projects/tir1/users/yihengz1/negotiation_robot/finite_state_machine/wfst_train/wfst_output/intents_0.5.wfst"):
        self.state_embedding = dict()
        self.current_state = 0
        self.transitions = dict()

        #initialize wfst
        print ("Initializing WFST...")
        current = -1
        #import pdb;pdb.set_trace()
        got_idx2strategy = False
        seen_header = False
        curridx = 0
        self.idx2strategy = {}  # index 2 strategy mapping
        for line in open(filename).read().split("\n"):
            if line == '': continue
            if not seen_header:
                if line.startswith('EST_Header_End'):
                    seen_header = True
                continue
            line_split = line.split()
            if line.startswith("(("):
                current = line_split[0][2:]
                self.state_embedding[int(current)] = list()
                self.transitions[int(current)] = dict()
            elif line.startswith(")"):
                got_idx2strategy = True
                continue
            else:
                self.state_embedding[int(current)].append(float(line_split[-1][:-1]))
                self.transitions[int(current)][line_split[0][1:]] = int(line_split[-2])
                if not got_idx2strategy:
                    self.idx2strategy[curridx] = line_split[0][1:]
                    curridx += 1
        print ("Loaded "+ str(len(self.transitions))+ " states.")
        print ("Initialization complete.")

    #look up embedding of a state
    def look_up_state_embedding(self, state_index):
        if state_index not in self.state_embedding:
            print ("state:" + str(state_index) + " does not exist.")
            exit(0)
        return self.state_embedding[state_index]

    #step function returns the next state given action
    def step(self, action, curr_state = None):
        if curr_state != None:							# if simulate
            next_state = self.transitions[curr_state][action]
        else:									# general step
            next_state = self.transitions[self.current_state][action]
            self.current_state = next_state
        return next_state

    def get_strategy_from_index(self, strategy_idx):
        return self.idx2strategy[strategy_idx]

    def get_seq_till_next_utt(self, end_marker = '<end>'):
        '''
        This will step and choose values till the time it gets an end (next utt)
        end_marker is '<end>'
        This will not change the current state of the WFST
        '''
        seq_of_states = [] # will not include current state
        seq_of_strategies_chosen = []
        seq_of_embeddings = []
        curr_state = self.current_state
        while True:
            e = self.look_up_state_embedding(curr_state)
            seq_of_embeddings.append(np.array(e).reshape(1, -1))
            strategy_idx = np.random.choice(np.arange(len(e)), p = e/np.sum(e))
            strategy = self.get_strategy_from_index(strategy_idx)
            if strategy == end_marker:
                return seq_of_states, seq_of_strategies_chosen, seq_of_embeddings
            seq_of_strategies_chosen.append(strategy.replace('<','').replace('>',''))
            seq_of_states.append(self.current_state)
            curr_state = self.step(strategy, curr_state)

