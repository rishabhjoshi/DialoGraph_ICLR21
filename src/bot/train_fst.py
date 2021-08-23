# THIS IS YIHENG's ORIGINAL CODE FOR FeHED!
# Author : Yiheng Zhou


from models import *
from models_amt import NA_Classifier
from cocoa.core.dataset import Example
#from nltk.translate.bleu_score import sentence_bleu
from wfst import *
import numpy as np
import os, pdb
#sys.path.insert(0, "/projects/tir1/users/yihengz1/negotiation/evaluation/findfeatures/")
sys.path.insert(0, "/projects/tir1/users/rjoshi2/negotiation/yiheng_negotiation/evaluation/findfeatures/")
from dialog_acts_extractor import *
import time
from joblib import dump, load
from sklearn.cluster import KMeans
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import utils
import json
from transformers import BertTokenizer
import pickle
from collections import defaultdict as ddict
from sklearn.metrics import classification_report, f1_score, average_precision_score


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_of_strategies_filename', dest='bag_of_strategies_filename', type=str, help="Please specify bag_of_strategies_filename path.")
    parser.add_argument('--fst_path', dest='fst_path', type=str, help="Please specify DA fsm path.")
    parser.add_argument('--fst_bs_path', dest='fst_bs_path', type=str, help="Please specify bs fsm path.")
    parser.add_argument('--model_name', dest='model_name', type=str, help="Please specify model name.")
    parser.add_argument('--attn_model', dest='attn_model', type=str)
    parser.add_argument('--hidden_size', dest='hidden_size', type=int)
    parser.add_argument('--encoder_n_layers', dest='encoder_n_layers', type=int, default=2)
    parser.add_argument('--decoder_n_layers', dest='decoder_n_layers', type=int, default=2)
    parser.add_argument('--sen_encoder_n_layers', dest='sen_encoder_n_layers', type=int, default=2)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1)
    parser.add_argument('--checkpoint_iter', dest='checkpoint_iter', type=int, help="no check point")
    parser.add_argument('--corpus_name', dest='corpus_name', type=str, help="corpus name missing")
    parser.add_argument('--clip', dest='clip', type=float, default=50.0)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.00005)
    parser.add_argument('--decoder_learning_ratio', dest='decoder_learning_ratio', type=float, default=5.0)
    parser.add_argument('--n_iteration', dest='n_iteration', type=int, default=5383 * 20)
    parser.add_argument('--print_every', dest='print_every', type=int, default=5383)
    parser.add_argument('--save_every', dest='save_every', type=int, default=5383)
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=-1,
                        help='Set to -1 if stochastic behavior is desired.')
    parser.add_argument('--kmeans_model_path', dest='kmeans_model_path', type=str, help="Please specify kmeans model path.")

    #add boolean
    parser_group = parser.add_mutually_exclusive_group(required=False)
    
    #state embedding
    parser_group.add_argument('--state_embedding', dest='state_embedding',
                              action='store_true',
                              help="Whether to use state embedding.")
    parser_group.add_argument('--no_state_embedding', dest='state_embedding',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(state_embedding=False)

    #train
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--train', dest='train',
                              action='store_true',
                              help="specify train")
    parser_group.add_argument('--no_train', dest='train',
                              action='store_false',
                              help="specify train")
    parser.set_defaults(train=True)

    #fine act
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--use_bs', dest='use_bs',
                              action='store_true',
                              help="using bs")
    parser_group.add_argument('--no_use_bs', dest='use_bs',
                              action='store_false',
                              help="using bs")
    parser.set_defaults(use_bs=True)

    #pretrain embedding
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--pretrain_word_embed', dest='pretrain_word_embed',
                              action='store_true',
                              help="using pretrained word embedding")
    parser_group.add_argument('--no_pretrain_word_embed', dest='pretrain_word_embed',
                              action='store_false',
                              help="using pretrained word embedding")
    parser.set_defaults(pretrain_word_embed=True)

    #pretrain dc
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--use_pretrain_dc', dest='use_pretrain_dc',
                              action='store_true',
                              help="using pretrained dialog act classifier")
    parser_group.add_argument('--no_use_pretrain_dc', dest='use_pretrain_dc',
                              action='store_false',
                              help="using pretrained dialog act classifier")
    parser.set_defaults(use_pretrain_dc=True)

    #use dc as input
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--dc_input', dest='dc_input',
                              action='store_true',
                              help="using dc as input")
    parser_group.add_argument('--no_dc_input', dest='dc_input',
                              action='store_false',
                              help="using dc as input")
    parser.set_defaults(dc_input=True)

    #joint learn
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--joint', dest='joint',
                              action='store_true',
                              help="joint training")
    parser_group.add_argument('--no_joint', dest='joint',
                              action='store_false',
                              help="joint training")
    parser.set_defaults(joint=True)

    #da
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--use_da', dest='use_da',
                              action='store_true',
                              help="use_da training")
    parser_group.add_argument('--no_use_da', dest='use_da',
                              action='store_false',
                              help="use_da training")
    parser.set_defaults(use_da=True)

    #use bag of strategies as it is in seq2seq
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--use_raw_bs', dest='use_raw_bs',
                              action='store_true',
                              help="use_raw_bs training")
    parser_group.add_argument('--no_use_raw_bs', dest='use_raw_bs',
                              action='store_false',
                              help="use_raw_bs training")
    parser.set_defaults(use_raw_bs=False)

    #use bag of strategies as it is in seq2seq but with cluster
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--use_cluster_bs', dest='use_cluster_bs',
                              action='store_true',
                              help="use_cluster_bs training")
    parser_group.add_argument('--no_use_cluster_bs', dest='use_cluster_bs',
                              action='store_false',
                              help="no_use_cluster_bs training")
    parser.set_defaults(use_cluster_bs=False)

    # true joint
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--true_joint', dest='true_joint',
                              action='store_true',
                              help="true_joint training")
    parser_group.add_argument('--no_true_joint', dest='true_joint',
                              action='store_false',
                              help="true_joint training")
    parser.set_defaults(true_joint=False)


    return parser.parse_args()

#load arguments
args = parse_arguments()
#print args
print(args, flush=True)

# Configure models
use_cluster_bs = args.use_cluster_bs
true_joint = args.true_joint
bag_of_strategies_filename = args.bag_of_strategies_filename
use_raw_bs = args.use_raw_bs
with_state_embedding = args.state_embedding
fst = args.fst_path
fst_bs = args.fst_bs_path
model_name = args.model_name
train_flag = args.train
bs_flag = args.use_bs
pre_train = args.pretrain_word_embed
use_pretrain_dc = args.use_pretrain_dc
dc_input = args.dc_input
use_da = args.use_da
wfsm = wfst("/projects/tir1/users/rjoshi2/negotiation/yiheng_negotiation_robot/finite_state_machine/wfst_train/wfst_output/"+fst)
wfsm_bs = wfst("/projects/tir1/users/rjoshi2/negotiation/yiheng_negotiation_robot/finite_state_machine/wfst_train/wfst_output/"+fst_bs)
joint = args.joint
output_name = model_name + ".out"
attn_model = args.attn_model
save_dir = os.path.join("data", "save")
hidden_size = args.hidden_size
encoder_n_layers = args.encoder_n_layers
decoder_n_layers = args.decoder_n_layers
sen_encoder_n_layers = args.sen_encoder_n_layers
dropout = args.dropout
batch_size = args.batch_size
checkpoint_iter = args.checkpoint_iter
corpus_name = args.corpus_name
clip = args.clip
learning_rate = args.learning_rate
decoder_learning_ratio = args.decoder_learning_ratio
n_iteration = args.n_iteration
print_every = args.print_every
save_every = args.save_every
kmeans_model_path = args.kmeans_model_path
kmeans = load(kmeans_model_path)

#global variables
#PAD = 0
PAD = 0
START = 1
SOS = 2
EOS = 3
OFFER = 4
REJECT = 5
ACCEPT = 6
QUIT = 7
UNK = 8
state_visits = Counter()
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
teacher_forcing_ratio = 1.0

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
cls_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS]'))
sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[SEP]'))

#preprocessing
raw_clusters = json.load(open("raw_clusters"))
vocab = {}
terminate_tokens = {"<accept>", "<reject>", "<quit>", ACCEPT, REJECT, QUIT}

# recommendation2uniformstrategymapping = {'seller_neg_sentiment': 'neg_sentiment',
#  'seller_pos_sentiment': 'pos_sentiment',
#  'buyer_neg_sentiment': 'neg_sentiment',
#  'buyer_pos_sentiment': 'pos_sentiment',
#  'first_person_plural_count_seller': 'first_person_plural_count',
#  'first_person_singular_count_seller': 'first_person_singular_count',
#  'first_person_plural_count_buyer': 'first_person_plural_count',
#  'first_person_singular_count_buyer': 'first_person_singular_count',
#  'third_person_singular_seller': 'third_person_singular',
#  'third_person_plural_seller': 'third_person_plural',
#  'third_person_singular_buyer': 'third_person_singular',
#  'third_person_plural_buyer': 'third_person_plural',
#  'number_of_diff_dic_pos': 'number_of_diff_dic_pos',
#  'number_of_diff_dic_neg': 'number_of_diff_dic_neg',
#  'buyer_propose': 'propose',
#  'seller_propose': 'propose',
#  'hedge_count_seller': 'hedge_count',
#  'hedge_count_buyer': 'hedge_count',
#  'assertive_count_seller': 'assertive_count',
#  'assertive_count_buyer': 'assertive_count',
#  'factive_count_seller': 'factive_count',
#  'factive_count_buyer': 'factive_count',
#  'who_propose': 'who_propose',
#  'seller_trade_in': 'trade_in',
#  'personal_concern_seller': 'personal_concern',
#  'sg_concern': 'sg_concern',
#  'liwc_certainty': 'liwc_certainty',
#  'liwc_informal': 'liwc_informal',
#  'politeness_seller_please': 'politeness_please',
#  'politeness_seller_gratitude': 'politeness_gratitude',
#  'politeness_seller_please_s': 'politeness_please',
#  'ap_des': 'ap_des',
#  'ap_pata': 'ap_pata',
#  'ap_infer': 'ap_infer',
#  'family': 'family',
#  'friend': 'friend',
#  'politeness_buyer_please': 'politeness_please',
#  'politeness_buyer_gratitude': 'politeness_gratitude',
#  'politeness_buyer_please_s': 'politeness_please',
#  'politeness_seller_greet': 'politeness_greet',
#  'politeness_buyer_greet': 'politeness_greet',
#  '<start>': '<start>'}

def word_to_index(split):
    '''
    Sets up vocab if split is 'train'
    Also, sets up the dialogues utterances2ids. and bert2ids
    preprocessing input data
    convert to index
    Returns train_data.
    :return:
    '''
    # RJ - Takes threshold and a path
    dialogs = dict()
    scenarios = dict()
    dialogs_bert = dict()
    scenarios_bert = dict()

    #the current size of vocab
    total = len(vocab['word2id'])  # will be 0 here RJ (used for train only)

    for dialog in raw_data[split]:
        scene_index = list()
        #read scenario
        #raw_s.append(dialog["scenario"])
        scene = (dialog["scenario"]["kbs"][1]["item"]["Category"] + " " + " ".join(dialog["scenario"]["kbs"][1]["item"]["Description"])+ " " + dialog["scenario"]["kbs"][1]["item"]["Title"])                   # Scene is category + description + title RJ
        scene = utils.normalizeString(scene, {"price":None}, dialog["scenario"], normalize_price=False)
        scenarios_bert[dialog['uuid']] = cls_token_id + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(scene)) + sep_token_id
        
        for word in scene.split():
            if split == 'train':
                if word not in vocab['word2id']:
                    vocab['word2id'][word] = total
                    vocab['id2word'][total] = word
                    total += 1
                vocab['word2freq'][word] += 1
            scene_index.append(vocab['word2id'][word])
        scenarios[dialog["uuid"]] = scene_index

        #import pdb; pdb.set_trace()
        utterances = parse_example(Example.from_dict(dialog, Scenario), price_tracker, templates)[2:] # first two are <start> or something RJ
        dialogs[dialog["uuid"]] = [["START", START]]
        dialogs_bert[dialog['uuid']] = [["START"] + cls_token_id + tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<start>')) + sep_token_id]
        u = 0
        previous = None
        for uter in dialog["events"]:
            tmp_dict = utterances[u]
            flag = False
            #encode agent types
            if uter["agent"] == 0:
                uter_index = ["b"]
                uter_index_bert = ['b']
            else:
                uter_index = ["s"]
                uter_index_bert = ['s']
            #read messages
            if uter["action"] == "message":		# dict.lf is logical form RJ (intents)
                utt = utils.normalizeString(uter["data"], tmp_dict.lf.to_dict(), dialog["scenario"], normalize_price=False)
                uter_index_bert += cls_token_id + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utt)) + sep_token_id
                for word in utt.split():
                    if split == 'train':
                        if word not in vocab['word2id']:
                            vocab['word2id'][word] = total
                            vocab['id2word'][total] = word
                            total += 1
                        vocab['word2freq'][word] += 1
                    uter_index.append(vocab['word2id'][word])
                uter_index.append(EOS)
            elif uter["action"] == "offer":
                uter_index += [OFFER, EOS]
                uter_index_bert += cls_token_id + tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<offer>")) + sep_token_id
                flag = True
            elif uter["action"] == "accept":
                uter_index += [ACCEPT, EOS]
                uter_index_bert += cls_token_id + tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<accept>")) + sep_token_id
                flag = True
            elif uter["action"] == "reject":
                uter_index += [REJECT, EOS]
                uter_index_bert += cls_token_id + tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<reject>")) + sep_token_id
                flag = True
            elif uter["action"] == "quit":
                uter_index += [QUIT, EOS]
                uter_index_bert += cls_token_id + tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<quit>")) + sep_token_id
                flag = True
            else:
                print ("unknown action from raw data!")
                exit()
            # if not flag:
            #     dialogs[dialog["uuid"]].append(uter_index)
            # else:
            #     if previous and previous["agent"] == uter["agent"]:
            #         dialogs[dialog["uuid"]][-1] = dialogs[dialog["uuid"]][-1][:-1] + uter_index[1:]
            #     else:
            #         dialogs[dialog["uuid"]].append(uter_index)
            dialogs[dialog["uuid"]].append(uter_index)
            dialogs_bert[dialog['uuid']].append(uter_index_bert)
            previous = uter
            u += 1
    if split == 'train':
        vocab['vocab_size'] = total
    data = {
        'utterances'     : dialogs,
        'scenarios'      : scenarios,
        'utterances_bert': dialogs_bert,
        'scenarios_bert' : scenarios_bert
    }
    return data

def word_to_index_eval(sentence):
    uter_index = list()
    uter_index_bert = list()
    if sentence.startswith("<offer>"):
        uter_index += [OFFER, EOS]
        uter_index_bert += cls_token_id + tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<offer>")) + sep_token_id
    elif sentence.startswith("<accept>"):
        uter_index += [ACCEPT, EOS]
        uter_index_bert += cls_token_id + tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<accept>")) + sep_token_id
    elif sentence.startswith("<reject>"):
        uter_index += [REJECT, EOS]
        uter_index_bert += cls_token_id + tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<reject>")) + sep_token_id
    elif sentence.startswith("<quit>"):
        uter_index += [QUIT, EOS]
        uter_index_bert += cls_token_id + tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<quit>")) + sep_token_id
    else:
        uter_index_bert = cls_token_id + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)) + sep_token_id
        for word in sentence.split():
            if word not in vocab['word2id']:
                uter_index.append(UNK)
            else:
                uter_index.append(vocab['word2id'][word])
        uter_index.append(EOS)

    return uter_index, uter_index_bert

def index_to_word(uter_index, scenario, is_bert = False):
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
    if is_bert:
        result = tokenizer.convert_ids_to_tokens(uter_index)
    else:
        #pdb.set_trace()
        for i in uter_index:
            if vocab['id2word'][i].startswith("<price>_"):
                result.append(str(float(vocab['id2word'][i].replace("<price>_","").replace(",","")) * (price-target) + target))
            else:
                result.append(vocab['id2word'][i])
    return result

from match_coef_and_ns import *
coef_weights = construct_weight_vector()

def get_encoder_input(utterances_embeddings, i, raw_bag_of_strategy, embedded_scene, models, params):# isval = False):
    '''
    :param utterances_embeddings: 1 x num_words x 600
    :param i: iteration i
    :param raw_bag_of_strategy:
    :param embedded_scene: scene embedded
    :param models:
    :param params:
    # :param isval: this is for teacher forced strategies.
    :return: encoder_input
    :return: strategies_input which is the new predicted bag of strategies
    '''
    # just_bag_of_strategies = True

    #pdb.set_trace()
    # ## Create the sequence inputs
    # strategies_encodings = list() # TODO CACHE AND MAKE SUPER FAST
    # for j in range(i+1):
    #     if j == 0:
    #         strategies_inp = torch.FloatTensor([1.0/len(raw_bag_of_strategy[0])] * len(raw_bag_of_strategy[0])).to(device).view(batch_size, -1)
    #     else:
    #         strategies_inp = torch.FloatTensor(raw_bag_of_strategy[:j]).to(device)
    #     if just_bag_of_strategies:
    #         strategies_enc = torch.FloatTensor(raw_bag_of_strategy[j+1][:-1]).to(device).view(batch_size, -1) # -1 because no start
    #     else:
    #         strategies_enc = models['strategy_encoder'](strategies_inp)
    #     # strategies_enc = strategies_inp
    #     # if strategies_enc.shape[0] != 1:
    #     #     strategies_enc = strategies_enc[-1].view(batch_size, -1)
    #     strategies_encodings.append(strategies_enc)
    # strategies_encodings = torch.stack(strategies_encodings, dim = 0) # size i x 1 x num_strat
    # # strategies_input = raw_bag_of_strategy[:i+1] # bs x num_utt x num_strats
    # # strategies_encoding = models['strategy_encoder'](strategies_input) # num_utt x num_strats (logits for prediction of next)

    batch_size = params['batch_size']
    # construct seq2seq input
    se_encoder_outputs = list()
    se_encoder_hidden = None
    #pdb.set_trace()
    for j in range(i + 1):
        se_encoder_hidden = None
        # pdb.set_trace()
        # if j == 0:
        #     tmp_input = np.reshape(input_variable[j][1], (batch_size, -1))
        # else:
        #     tmp_input = np.reshape(input_variable[j][1:], (batch_size, -1))
        if j < len(raw_bag_of_strategy):
            raw_bag_of_strategy_indexed = raw_bag_of_strategy[j]
        else:
            raw_bag_of_strategy_indexed = raw_bag_of_strategy[-1]

        ## If not seq2seq
        # see if need to give embedding
        #pdb.set_trace() ###CHECK
        # se_encoder_output, se_encoder_hidden = models['dial_encoder'](torch.LongTensor(tmp_input).to(device), se_encoder_hidden)

        #se_encoder_output = torch.max(se_encoder_output, dim=1)[0]
        se_encoder_output = se_encoder_output.unsqueeze(0)
        se_tmp = torch.cat((
            #torch.FloatTensor([1.0 / len(raw_bag_of_strategy_indexed)] * len(raw_bag_of_strategy_indexed)).to(device).view(batch_size, -1),
            se_encoder_output[-1],
            embedded_scene
        ), dim = 1)
        se_encoder_outputs.append(se_tmp) # just the utterance
    se_encoder_outputs = torch.stack(se_encoder_outputs, dim=0)
    se_encoder_outputs = torch.cat((se_encoder_outputs, strategies_encodings), dim=2)
    return se_encoder_outputs, strategies_inputs[-1] # sentences x 1 x 623, predicted_strategies

na_loss_fnc = torch.nn.BCEWithLogitsLoss()

def get_strat_embedding(datapoint, i, models, params):
    '''

    :param datapoint:
    :param i:
    :param models:
    :param params:
    :return:
    '''
    strat_embedding = datapoint['strategies_vec'][:i+1][:, :-1] # remove start
    strat_embedding = torch.FloatTensor(strat_embedding).to(device)
    #target = torch.FloatTensor(datapoint['strategies_vec'][1:i+2][:, :-1]).to(device).view(1, i+2-1, -1)
    target = torch.FloatTensor(datapoint['strategies_vec'][i+1:i+2][:, :-1]).to(device).view(1, -1)
    strat_pred, loss = models['strat_encoder'](strat_embedding, is_strat = True, target=target)
    return strat_pred, loss # 1 x num_strat
    #return strat_embedding[-1, :].view(1, -1)

def get_da_embedding(datapoint, i, models, params):
    '''
    :param datapoint:
    :param i:
    :param models:
    :param params:
    :return:
    '''
    dial_acts = datapoint['dial_acts_vec'][:i+1]
    dial_vec = np.zeros((i+1, params['num_da']))
    dial_acts[dial_acts == params['num_da']] = 0
    dial_vec[np.arange(dial_acts.size), dial_acts] = 1
    dial_embedding = torch.FloatTensor(dial_vec).to(device)
    #target = torch.LongTensor(datapoint['dial_acts_vec'][1:i+2]).to(device)#.view(1, i+2-1, -1)#.view(1, -1)
    target = torch.LongTensor(datapoint['dial_acts_vec'][i+1:i+2]).to(device)  # .view(1, i+2-1, -1)#.view(1, -1)
    dial_pred, loss = models['da_encoder'](dial_embedding, is_strat = False, target =target)
    return dial_pred, loss # 1 x num_da
    #return dial_embedding[-1, :].view(1, -1)

def train_rj(datapoint, models, optimizers, embedding, criteria, params, to_train='all'):
    print_losses = []
    print_dc_losses = []
    print_na_losses = []
    n_totals = 0
    n_dc_totals = 0
    n_na_totals = 0
    length = len(datapoint['toks_space'])
    batch_size = params['batch_size']

    # initialize wfsm
#    wfsm.current_state = 0
#    wfsm_bs.current_state = 0

    # no communication between each dialog
    da_state_embeddings = []
    bs_state_embeddings = []

    raw_bag_of_strategy = datapoint['strategies_vec']
    input_variable = datapoint['toks_space']
    #raw_bag_of_strategy.insert(0, [0.0] * len(raw_bag_of_strategy[0]))  # zero removed as I have put that anyway.

    #embedded_scene = embedding(torch.LongTensor(datapoint['scene_space'][:]).to(device))  # NO EOS in scene
    #embedded_scene = torch.mean(embedded_scene.t(), dim=1).reshape(1, -1)
    embedded_scene, _ = models['dial_encoder'](torch.LongTensor(datapoint['scene_space'][1:]).to(device).reshape(batch_size, -1)) # 1 x 600

    utterances_embeddings = list()
    for i in range(length):
        utt_toks = torch.LongTensor(np.reshape(input_variable[i][1:], (batch_size, -1)))
        utt_embed, _ = models['dial_encoder'](torch.LongTensor(utt_toks).to(device), None) # 1 x 1 x 300
        utterances_embeddings.append(utt_embed) # take the SEP encoding bs x 600
    utterances_embeddings = torch.stack(utterances_embeddings, dim=1) # 1 x num_utt x 600

    for i in range(length-1):
        loss = 0
        dc_loss = 0
        na_loss = 0

        # Zero gradients
        optimizers['utt_encoder'].zero_grad()
        optimizers['dial_encoder'].zero_grad()
        #if to_train == 'only_da':
        optimizers['da_classifier'].zero_grad()
        optimizers['da_encoder'].zero_grad()
        #elif to_train == 'only_strat':
        optimizers['strat_classifier'].zero_grad()
        optimizers['strat_encoder'].zero_grad()
        #else:
        optimizers['utt_decoder'].zero_grad()

        # lookup state embedding #TODO STEP AT END
        bs_state_embedding, strat_loss = get_strat_embedding(datapoint, i, models, params)
        da_state_embedding, da_loss = get_da_embedding(datapoint, i, models, params)
        bs_state_embeddings.append(bs_state_embedding)
        da_state_embeddings.append(da_state_embedding)

        # if to_train == 'only_da':
        #     #dc_output = models['da_classifier'](encoder_outputs[-1]) # only take EOS
        #     #dc_output = da_state_embedding
        #     try:
        #         #mask_loss = criteria(dc_output, target_variable[0])
        #         #pdb.set_trace()
        #         mask_loss = da_loss
        #         loss += mask_loss
        #         print_losses.append(mask_loss.item())
        #         n_totals += 1
        #     except:
        #         pdb.set_trace()
        # elif to_train == 'only_strat':
        #     ### na_output = models['strat_classifier'](encoder_outputs[-1])
        #     #pdb.set_trace()
        #     #na_output = bs_state_embedding
        #     #na_output = torch.sigmoid(na_output)
        #     #mask_loss = na_loss_fnc(na_output, target_variable)
        #     mask_loss = strat_loss
        #     loss += mask_loss
        #     print_losses.append(mask_loss.item())
        #     n_totals += 1
        #
        # if to_train == 'only_da':
        #     _ = torch.nn.utils.clip_grad_norm_(models['da_encoder'].parameters(), clip)
        #     optimizers['da_encoder'].step()
        #     loss.backward(retain_graph=True)
        # elif to_train == 'only_strat':
        #     _ = torch.nn.utils.clip_grad_norm_(models['strat_encoder'].parameters(), clip)
        #     optimizers['strat_encoder'].step()
        #     loss.backward(retain_graph=True)
        #
        # loss = 0

        # skip buyer or seller
        if datapoint['agent_list'][i+1] != 1:
        #if input_variable[i + 1][0] != "s":
            continue

        encoder_input = utterances_embeddings[:, :i+1, :]
        encoder_input = torch.cat((encoder_input, embedded_scene.unsqueeze(1).repeat(1, i+1,1)), dim=2)

        # expand encoder input using states
        #pdb.set_trace()
        encoder_input = torch.cat((encoder_input, torch.stack(bs_state_embeddings, dim=1)), dim = 2)
        encoder_input = torch.cat((encoder_input, torch.stack(da_state_embeddings, dim=1)), dim = 2)
        encoder_input = encoder_input.permute(1, 0, 2)
        encoder_outputs, encoder_hidden = models['utt_encoder'](encoder_input)  # num_words x 1 x 300, 4 x 1 x 300

        #pdb.set_trace()
        # if to_train == 'only_da':
        #     target_variable = np.reshape(datapoint['dial_acts_vec'][i+1], (-1, batch_size))
        #     target_variable = torch.LongTensor(target_variable).to(device)
        # elif to_train == 'only_strat':
        #     target_variable = np.reshape(raw_bag_of_strategy[i + 1][:-1], (batch_size, -1)) # ignore start
        #     target_variable = torch.FloatTensor(target_variable).to(device)
        # else:
        #     #pdb.set_trace()
        target_variable = np.reshape(input_variable[i + 1][1:], (-1, batch_size)) # skip SOS
        target_variable = torch.LongTensor(target_variable).to(device)

        # if to_train == 'only_da':
        #dc_output = models['da_classifier'](encoder_outputs[-1]) # only take EOS
        #dc_output = da_state_embedding
        try:
            #mask_loss = criteria(dc_output, target_variable[0])
            mask_loss = da_loss
            dc_loss += mask_loss
            print_dc_losses.append(mask_loss.item())
            n_dc_totals += 1
        except:
            pdb.set_trace()
        #loss.backward(retain_graph=True)
        #elif to_train == 'only_strat':
        ### na_output = models['strat_classifier'](encoder_outputs[-1])
        #pdb.set_trace()
        #na_output = bs_state_embedding
        #na_output = torch.sigmoid(na_output)
        #mask_loss = na_loss_fnc(na_output, target_variable)
        mask_loss = strat_loss
        na_loss += mask_loss
        print_na_losses.append(mask_loss.item())
        n_na_totals += 1
        #loss.backward(retain_graph=True)
        #elif to_train == 'decoder':
        #with torch.no_grad():
        dc_output = da_state_embedding# models['da_classifier'](encoder_outputs[-1])
        ### na_output = models['strat_classifier'](encoder_outputs[-1])
        na_output = bs_state_embedding
        #na_output = torch.sigmoid(na_output)
        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)
        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:models['utt_decoder'].n_layers]
        # Forward batch of sequences through decoder one time step at a time
        decoder_outputs = list()
        #pdb.set_trace()
        for t in range(len(target_variable)):
            #pdb#.set_trace()
            decoder_output, decoder_hidden = models['utt_decoder'](
                decoder_input, decoder_hidden, encoder_outputs, dc_output, torch.sigmoid(na_output)
            )
            _, topi = decoder_output.topk(1)
            decoder_outputs.append(topi[0][0].item())
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            #tpdb.set_trace()
            mask_loss = criteria(decoder_output, target_variable[t])
            # Calculate and accumulate loss
            loss += mask_loss
            print_losses.append(mask_loss.item())
            n_totals += 1

        # Perform backpropatation
        loss += na_loss + dc_loss
        try:
            loss.backward(retain_graph = True)
        except:
            pdb.set_trace()

        # Clip gradients: gradients are modified in place
        _ = torch.nn.utils.clip_grad_norm_(models['utt_encoder'].parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(models['dial_encoder'].parameters(), clip)
        #if to_train == 'only_da':
        _ = torch.nn.utils.clip_grad_norm_(models['da_classifier'].parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(models['da_encoder'].parameters(), clip)
        #elif to_train == 'only_strat':
        _ = torch.nn.utils.clip_grad_norm_(models['strat_classifier'].parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(models['strat_encoder'].parameters(), clip)
        #elif to_train == 'decoder':
        _ = torch.nn.utils.clip_grad_norm_(models['utt_decoder'].parameters(), clip)


        # Adjust model weights
        optimizers['utt_encoder'].step()
        optimizers['dial_encoder'].step()
        #if to_train == 'only_da':
        optimizers['da_classifier'].step()
        optimizers['da_encoder'].step()
        #elif to_train == 'only_strat':
        optimizers['strat_classifier'].step()
        optimizers['strat_encoder'].step()
        #elif to_train == 'decoder':
        optimizers['utt_decoder'].step()

    if n_totals == 0:
        return -1
    return sum(print_losses) / n_totals, sum(print_dc_losses)/n_dc_totals, sum(print_na_losses)/n_na_totals


def joint_trainIters(train_data, dev_data, models, optimizers, embedding, criteria, params):
    # Initializations
    print('Initializing ...')
    indexes = np.arange(len(train_data))
    np.random.shuffle(indexes)
    # uuids = list(input_dialogs.keys())
    # np.random.shuffle(uuids)
    # training_uuids = uuids

    best_iter = -1
    best_loss = 100000
    start_iteration = 1
    print_dc_loss = 0
    print_loss = 0
    print_rl_loss = 0
    print_na_loss = 0

    for iteration in range(params['n_iteration']):
        print('Iteration : ', iteration)
        np.random.shuffle(indexes)
        for k, idx in enumerate(indexes):
            datapoint = train_data[idx]
            # dc_loss = train_rj(datapoint, models, optimizers, embedding, criteria, params, to_train='only_da') # train_dc
            # if dc_loss != -1:
            #     print_dc_loss += dc_loss
            #
            # strat_loss = train_rj(datapoint, models, optimizers, embedding, criteria, params, to_train='only_strat') #train_na
            # if strat_loss != -1:
            #     print_na_loss += strat_loss

            loss, dc_loss, strat_loss = train_rj(datapoint, models, optimizers, embedding, criteria, params, to_train='decoder') # train
            if dc_loss != -1:
                print_dc_loss += dc_loss
            if strat_loss != -1:
                print_na_loss += strat_loss
            if loss != -1:
                print_loss += loss
            #pdb.set_trace()
            if (k+1) % params['print_every'] == 0:
                print_loss_avg = print_loss / params['print_every']
                print_dc_loss_avg = print_dc_loss / params['print_every']
                print_rl_loss_avg = print_rl_loss / params['print_every']
                print_na_loss_avg = print_na_loss / params['print_every']
                print(
                    "Epoch: {}; Percent complete: {:.2f}%; Average loss: {:.4f}; Average dc loss: {:.4f}; Average rl loss: {:.4f}; Average na loss: {:.4f};".format(
                        iteration, (k / len(indexes)) * 100, print_loss_avg, print_dc_loss_avg, print_rl_loss_avg,
                        print_na_loss_avg))
                with open(output_name, "a") as output:
                    output.write(
                        "Epoch: {}; Percent complete: {:.2f}%; Average loss: {:.4f}; Average dc loss: {:.4f}; Average rl loss: {:.4f}; Average na loss: {:.4f};".format(
                            iteration, (k / len(indexes)) * 100, print_loss_avg, print_dc_loss_avg,
                            print_rl_loss_avg, print_na_loss_avg) + "\n")
                print_loss = 0
                print_dc_loss = 0
                print_rl_loss = 0
                print_na_loss = 0
                print_flag = 1

            if (k+1) % params['save_every'] == 0:
                print('Saving model')
                for model in models:
                    models[model].eval()
                with torch.no_grad():
                    eval_loss = evaluateIters_rj(dev_data, models, optimizers, embedding, criteria, params)
                for model in models:
                    models[model].train()

                if eval_loss < best_loss:
                    best_iter = iteration
                    best_loss = eval_loss
                directory = os.path.join(params['save_dir'], params['model_name'], params['corpus_name'],
                                         '{}-{}_{}'.format(params['encoder_n_layers'], params['decoder_n_layers'], params['hidden_size']))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'iteration': iteration,
                    'en': models['utt_encoder'].state_dict(),
                    'de': models['utt_decoder'].state_dict(),
                    'se': models['dial_encoder'].state_dict(),
                    "dc": models['da_classifier'].state_dict(),
                    "na": models['strat_classifier'].state_dict(),
                    'en_opt': optimizers['utt_encoder'].state_dict(),
                    'de_opt': optimizers['utt_decoder'].state_dict(),
                    'se_opt': optimizers['dial_encoder'].state_dict(),
                    "dc_opt": optimizers['da_classifier'].state_dict(),
                    "na_opt": optimizers['strat_classifier'].state_dict(),
                    'loss': loss,
                    'embedding': embedding.state_dict(),
                    #'cluster_embeddings': cluster_embeddings.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
                save_flag = 1
    print("Training Complete")
    print("The best model is " + str(best_iter) + " with loss " + str(best_loss))
    print("states visit:")
    print(state_visits)
    # for iteration in range(start_iteration, n_iteration + 1):
    #     uuid = training_uuids[(iteration - 1)%len(training_uuids)]
    #     input_variable = input_dialogs[uuid] #datapoint['toks_space']
    #     input_variable_seq = seq_of_intents_train[uuid][2:]
    #
    #     # RL
    #     # if iteration >= 53800:
    #     #     rl_loss = train_with_rl(uuid, seq_of_fine_intents_raw[uuid.lower()], cluster_embeddings, raw_bag_of_strategies[uuid.lower()], seq_of_fine_intents[uuid.lower()], scenario_index[uuid], input_variable, input_variable_seq, encoder, decoder, sen_encoder, embedding, encoder_optimizer, decoder_optimizer, sen_ecoder_optimizer, batch_size, clip, criteria, d_classifier, na_classifier, na_classifier_optimizer)
    #     #     if rl_loss != -1:
    #     #         print_rl_loss += rl_loss
    #
    #     # Run a training iteration with batch
    #     dc_loss = train_dc(cluster_embeddings, raw_bag_of_strategies[uuid.lower()], seq_of_fine_intents[uuid.lower()], scenario_index[uuid], input_variable, input_variable_seq, encoder, d_classifier, sen_encoder, embedding, encoder_optimizer, d_classifier_optimizer, sen_ecoder_optimizer, batch_size, clip, criteria)
    #     if dc_loss != -1:
    #         print_dc_loss += dc_loss
    #
    #     # Run a training iteration with batch
    #     na_loss = train_na(cluster_embeddings, raw_bag_of_strategies[uuid.lower()], seq_of_fine_intents[uuid.lower()], scenario_index[uuid], input_variable, input_variable_seq, encoder, sen_encoder, embedding, encoder_optimizer, sen_ecoder_optimizer, batch_size, clip, criteria, na_classifier, na_classifier_optimizer)
    #     if na_loss != -1:
    #         print_na_loss += na_loss
    #
    #     # Run a training iteration with batch
    #     loss = train(cluster_embeddings, raw_bag_of_strategies[uuid.lower()], seq_of_fine_intents[uuid.lower()], scenario_index[uuid], input_variable, input_variable_seq, encoder, decoder, sen_encoder, embedding, encoder_optimizer, decoder_optimizer, sen_ecoder_optimizer, batch_size, clip, criteria, d_classifier, na_classifier, na_classifier_optimizer)
    #     if loss != -1:
    #         print_loss += loss
    #
    #     # Print progress
    #     if iteration % print_every == 0:
    #         #shuffle training examples
    #         np.random.shuffle(training_uuids)
    #         print_loss_avg = print_loss / print_every
    #         print_dc_loss_avg = print_dc_loss/ print_every
    #         print_rl_loss_avg = print_rl_loss/ print_every
    #         print_na_loss_avg = print_na_loss/print_every
    #         print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}; Average dc loss: {:.4f}; Average rl loss: {:.4f}; Average na loss: {:.4f};".format(iteration, iteration / n_iteration * 100, print_loss_avg, print_dc_loss_avg, print_rl_loss_avg, print_na_loss_avg))
    #         with open(output_name, "a") as output:
    #             output.write("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}; Average dc loss: {:.4f}; Average rl loss: {:.4f}; Average na loss: {:.4f};".format(iteration, iteration / n_iteration * 100, print_loss_avg, print_dc_loss_avg, print_rl_loss_avg, print_na_loss_avg) + "\n")
    #         print_loss = 0
    #         print_dc_loss = 0
    #         print_rl_loss = 0
    #         print_na_loss = 0
    #
    #     # Save checkpoint
    #     if (iteration % save_every == 0):
    #         encoder.eval()
    #         decoder.eval()
    #         sen_encoder.eval()
    #         d_classifier.eval()
    #         na_classifier.eval()
    #         with torch.no_grad():
    #             eval_loss = evaluateIters(cluster_embeddings, raw_bag_of_strategies, seq_of_fine_intents_raw, seq_of_fine_intents, dev_scenario_index, dev_dialogs, seq_of_intents_dev, encoder, decoder, sen_encoder, embedding, criteria, d_classifier, na_classifier, na_classifier_optimizer)
    #             eval_dc_loss = evaluate_dc_Iters(cluster_embeddings, raw_bag_of_strategies, seq_of_fine_intents, dev_scenario_index, dev_dialogs, seq_of_intents_dev, encoder, d_classifier, sen_encoder, embedding, criteria)
    #         encoder.train()
    #         decoder.train()
    #         sen_encoder.train()
    #         d_classifier.train()
    #         na_classifier.train()
    #         if eval_loss < best_loss:
    #             best_iter = iteration
    #             best_loss = eval_loss
    #         directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
    #         if not os.path.exists(directory):
    #             os.makedirs(directory)
    #         torch.save({
    #             'iteration': iteration,
    #             'en': encoder.state_dict(),
    #             'de': decoder.state_dict(),
    #             'se': sen_encoder.state_dict(),
    #             "dc": d_classifier.state_dict(),
    #             "na": na_classifier.state_dict(),
    #             'en_opt': encoder_optimizer.state_dict(),
    #             'de_opt': decoder_optimizer.state_dict(),
    #             'se_opt': sen_ecoder_optimizer.state_dict(),
    #             "dc_opt": d_classifier_optimizer.state_dict(),
    #             "na_opt": na_classifier_optimizer.state_dict(),
    #             'loss': loss,
    #             'embedding': embedding.state_dict(),
    #             'cluster_embeddings': cluster_embeddings.state_dict()
    #         }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
    #
    # print ("Training Complete")
    # print ("The best model is " + str(best_iter) + " with loss " + str(best_loss))
    # print ("states visit:")
    # print (state_visits)

def evaluateIters_rj(dev_data, models, optimizers, embedding, criteria, params, test_eval = False):
    print_loss = 0.0
    print_loss_u = 0.0
    print_loss_d = 0.0
    d_accs = Counter()
    d_acc_totals = Counter()
    labels_acc = Counter()
    labels_total = Counter()
    total = 0
    find_errors = dict()
    fine_acts_accs = Counter()
    fine_acts_totals = Counter()
    bigram_fine_acts_accs = Counter()
    bigram_fine_acts_totals = Counter()
    bleu_scores = []
    TPs = 0.0
    FPs = 0.0
    TNs = 0.0
    FNs = 0.0
    na_preds = []
    na_labels = []
    na_logits = []
    for k, datapoint in enumerate(dev_data):
    #for uuid in dev_dialogs:
        loss, u_loss, d_loss, metrics, na_preds_one, na_labels_one = evaluate_rj(datapoint, models, optimizers, embedding, criteria, params, test_eval)

        d_acc, d_acc_total, label_acc, label_total, fine_acts_acc, fine_acts_total, bigram_fine_acts_acc, bigram_fine_acts_total, bleu_score, TP, FP, TN, FN, = \
            metrics['d_acc'], metrics['d_acc_total'], metrics['label_acc'], metrics['label_acc_total'], metrics['fine_acts_acc'], metrics['fine_acts_total'], \
            metrics['bigram_fine_acts_acc'], metrics['bigram_fine_acts_total'], metrics['bleu_score'], metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN']

        if loss != -1 :#or len(na_labels_one) == 0: # na_labels_0 is for troublesome cases
            TPs += TP
            FPs += FP
            TNs += TN
            FNs += FN
            bleu_scores += bleu_score
            fine_acts_accs += fine_acts_acc
            fine_acts_totals += fine_acts_total
            bigram_fine_acts_accs += bigram_fine_acts_acc
            bigram_fine_acts_totals += bigram_fine_acts_total
            print_loss += loss
            print_loss_u += u_loss
            print_loss_d += d_loss
            total += 1
            d_accs += d_acc
            d_acc_totals += d_acc_total
            labels_acc += label_acc
            labels_total += label_total
            #pdb.set_trace()
            try:
                if len(na_labels_one) == 0:
                    pdb.set_trace()
                #if len(na_labels_one) != 0:
                na_logits.append(np.array([k.cpu().detach().numpy() for k in na_preds_one]))
                na_preds.append(np.array([torch.sigmoid(k).cpu().detach().numpy() for k in na_preds_one]))
                na_labels.append(np.array(na_labels_one))
                # na_preds += np.array([k.cpu().detach().numpy() for k in na_preds_one])
                # na_labels += np.array(na_labels_one)
            except:
                pdb.set_trace()  # TODO SKIPS FOR THREE
        else:
            pdb.set_trace()

        if (k + 1) % params['print_every'] == 0:
            print("Eval : Percent complete: {:.2f}%; Average loss: {:.4f};".format((k / len(dev_data)) * 100, print_loss / total))

    # import pdb;
    # pdb.set_trace()
    #     find_errors[uuid] = loss

    # with open("find_errors_20", "w") as output:
    #     json.dump(find_errors, output)

    precision = TPs / (TPs + FPs)
    recall = TPs / (TPs + FNs)
    #micro_F1 = 2 * precision * recall / max(1, (precision + recall))
    micro_F1 = 2 * precision * recall / (precision + recall)
    preds_all = np.concatenate(na_preds, 0)
    logits_all = np.concatenate(na_logits, 0)
    preds_all_y = copy.deepcopy(preds_all)
    preds_all_y[preds_all_y >= 0.5] = 1
    preds_all_y[preds_all_y < 0.5] = 0
    labels_all = np.concatenate(na_labels, 0)
    #pdb.set_trace()
    macro_F1 = f1_score(labels_all, preds_all_y, average='macro')
    micro_f1 = f1_score(labels_all, preds_all_y, average='micro')
    aps      = average_precision_score(labels_all.reshape(-1), logits_all.reshape(-1))#, average='macro')
    assert micro_f1 == micro_F1
    #accuracyy = (TPs + TNs) / max(1, (TPs + FPs + FNs + TNs))
    print (classification_report(labels_all, preds_all_y))
    with open(output_name, "a") as output:
        output.write("Total Loss on Dev set is: " + str(print_loss / total) + "\n")
    print("Total Loss per word on Dev set is: " + str(print_loss / total))
    print("Total ACC for fine act: " + str(sum(fine_acts_accs.values()) / sum(fine_acts_totals.values())))
    print(fine_acts_accs)
    print(fine_acts_totals)
    for key in fine_acts_accs:
        print(key, fine_acts_accs[key] / fine_acts_totals[key])
    print("Total ACC for bigram fine act: " + str(
        sum(bigram_fine_acts_accs.values()) / sum(bigram_fine_acts_totals.values())))
    print(bigram_fine_acts_accs)
    print(bigram_fine_acts_totals)
    for key in bigram_fine_acts_accs:
        print(key, bigram_fine_acts_accs[key] / bigram_fine_acts_totals[key])
    print("BLEU:", sum(bleu_scores) / len(bleu_scores))
    print("Micro F1:", micro_F1)
    print("Macro F1:", macro_F1)
    print("Average Precision Score:", aps)
    #print("ACC:", accuracyy)
    # print ("Total Loss per utt on Dev set is: " + str(print_loss_u/total))
    # print ("Total Loss per dialog on Dev set is: " + str(print_loss_d/total))
    # for act in d_accs:
    #     print ("act:", act)
    #     print ("acc:", d_accs[act]/d_acc_totals[act])
    # print (d_acc_totals)
    # for intent in labels_acc:
    #     print (intent, labels_acc[intent]/labels_total[intent], labels_total[intent])
    # print ("label total acc:", sum(labels_acc.values())/sum(labels_total.values()))
    return print_loss / total

from construct_ns_bigram import *
#ns_bigram_prob, ns_bigram_raw = ns_bigram()
ns_bigram, _ = ns_bigram_as_a_whole()
import copy

def evaluate_rj(datapoint, models, optimizers, embedding, criteria, params, test_eval):
    # Set device options
    # Initialize variables

    print_losses = []
    n_totals = 0
    u_totals = 0
    length = len(datapoint['toks_space'])
    batch_size = params['batch_size']
    #raw_bag_of_strategy.insert(0, [0.0] * len(raw_bag_of_strategy[0]))
    raw_bag_of_strategy = datapoint['strategies_vec']
    input_variable = datapoint['toks_space']
    raw_example = copy.deepcopy(datapoint['raw_data'])
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0

    #embedded_scene = embedding(torch.LongTensor(datapoint['scene_space'][:]).to(device))  # NO EOS in scene
    embedded_scene, _ = models['dial_encoder'](torch.LongTensor(datapoint['scene_space'][1:]).to(device).reshape(batch_size, -1))  # 1 x 600

    utterances_embeddings = list()
    for i in range(length):
        utt_toks = torch.LongTensor(np.reshape(input_variable[i][1:], (batch_size, -1)))
        utt_embed, _ = models['dial_encoder'](torch.LongTensor(utt_toks).to(device), None)  # 1 x 1 x 300
        utterances_embeddings.append(utt_embed)  # take the SEP encoding bs x 600
    utterances_embeddings = torch.stack(utterances_embeddings, dim=1)  # 1 x num_utt x 600

    # utterances = parse_example(Example.from_dict(raw_example, Scenario), price_tracker, templates)[2:]

    # print (raw_example["events"])
    # for i in range(len(raw_example)):
    #     print (utterances[i].lf.to_dict())

    # initialize wfsm
#    wfsm.current_state = 0
#    wfsm_bs.current_state = 0

    # no communication between each dialog
    se_encoder_outputs = list()
    da_state_embeddings = []
    bs_state_embeddings = []
    bleu_scores = []

    na_preds = []
    na_labels = [] # TODO Dont add strategies / predict if at the end agent -1
    # Forward pass through encoder
    for i in range(length - 1):
        loss = 0

        # lookup state embedding #TODO STEP AT END
        bs_state_embedding, _ = get_strat_embedding(datapoint, i, models, params)
        da_state_embedding, _ = get_da_embedding(datapoint, i, models, params)
        bs_state_embeddings.append(bs_state_embedding)
        da_state_embeddings.append(da_state_embedding)

        # skip buyer or seller
        if datapoint['agent_list'][i + 1] != 1:
            continue

        encoder_input = utterances_embeddings[:, :i + 1, :]
        encoder_input = torch.cat((encoder_input, embedded_scene.unsqueeze(1).repeat(1, i + 1, 1)), dim=2)

        # expand encoder input using states
        # pdb.set_trace()
        #pdb.set_trace()
        encoder_input = torch.cat((encoder_input, torch.stack(bs_state_embeddings, dim=1)), dim=2)
        encoder_input = torch.cat((encoder_input, torch.stack(da_state_embeddings, dim=1)), dim=2)
        encoder_input = encoder_input.permute(1, 0, 2)
        encoder_outputs, encoder_hidden = models['utt_encoder'](encoder_input)  # num_words x 1 x 300, 4 x 1 x 300

        #pdb.set_trace()
        target_variable = np.reshape(input_variable[i + 1][1:], (-1, batch_size)) # skip sos
        target_variable = torch.LongTensor(target_variable).to(device)


        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)
        dc_output = da_state_embedding#models['da_classifier'](encoder_outputs[-1])
        na_output = bs_state_embedding#models['strat_classifier'](encoder_outputs[-1])
        #na_output = torch.sigmoid(na_output)

        #####F1 for SP#####
        #pdb.set_trace()
        #num_end_ignore = np.sum(np.array(datapoint['agent_list']) == -1) - 1 # -1 for start
        if i + 1 < len(raw_bag_of_strategy): # Dont consider last
            #pdb.set_trace()
            # ignore -1 ones
            if datapoint['agent_list'][i+1] == -1: continue
            na_preds.append(na_output[0])
            na_labels.append(raw_bag_of_strategy[i + 1][:-1]) # ignore start
            for na_i, na_o in enumerate(torch.sigmoid(na_output[0])):
                if na_o >= 0.5 and raw_bag_of_strategy[i + 1][na_i] == 1:
                    TP += 1
                elif na_o < 0.5 and raw_bag_of_strategy[i + 1][na_i] == 1:
                    FN += 1
                elif na_o >= 0.5 and raw_bag_of_strategy[i + 1][na_i] == 0:  # TODO WHAT IS THIS BAKCHODI!!!
                    FP += 1
                else:
                    TN += 1
                # if na_o >= 0.000005 and raw_bag_of_strategy[i+1][na_i] == 1:
                #     TP += 1
                # elif na_o < 0.000005 and raw_bag_of_strategy[i+1][na_i] == 1:
                #     FN += 1
                # elif na_o >= 0.000005 and raw_bag_of_strategy[i+1][na_i] == 0:  # TODO WHAT IS THIS BAKCHODI!!!
                #     FP += 1
                # else:
                #     TN += 1

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:models['utt_decoder'].n_layers]        # TODO CHECK IF LAST LAYER?? 2x1x300 <- 4x1x300

        decoder_outputs = list()
        eos_flag = False
        # Forward batch of sequences through decoder one time step at a time
        for t in range(len(target_variable)):
            decoder_output, decoder_hidden = models['utt_decoder'](
                decoder_input, decoder_hidden, encoder_outputs, dc_output, torch.sigmoid(na_output)
            )
            # noo teacher forcing
            _, topi = decoder_output.topk(1)
            #pdb.set_trace()
            if not eos_flag:
                decoder_outputs.append(topi[0][0].item())
                if decoder_outputs[-1] == EOS:
                    eos_flag = True
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)

            # change
            # decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # decoder_input = torch.unsqueeze(decoder_input, 0)
            # change

            # Calculate and accumulate loss
            mask_loss = criteria(decoder_output, target_variable[t])
            loss += mask_loss
            print_losses.append(mask_loss.item())
            n_totals += 1
        u_totals += 1

        # decode outputs
        decoded_words = " ".join(index_to_word(decoder_outputs, raw_example["scenario"])[:-1])
        #print(decoded_words)
        ref_words = ""

        if raw_example["events"][i]["action"] == "accept":
            ref_words = "<accept>"
        elif raw_example["events"][i]["action"] == "reject":
            ref_words = "<reject>"
        elif raw_example["events"][i]["action"] == "offer":
            ref_words = "<offer>"
        elif raw_example["events"][i]["action"] == "message":
            ref_words = raw_example["events"][i]["data"]
        elif raw_example["events"][i]["action"] == "quit":
            ref_words = "<quit>"

        if "<accept>" in decoded_words:
            raw_example["events"][i] = {"data": None, "action": "accept", "agent": 1, "time": time.time(),
                                        "start_time": time.time()}
        elif "<reject>" in decoded_words:
            raw_example["events"][i] = {"data": None, "action": "reject", "agent": 1, "time": time.time(),
                                        "start_time": time.time()}
        elif "<quit>" in decoded_words:
            raw_example["events"][i] = {"data": None, "action": "quit", "agent": 1, "time": time.time(),
                                        "start_time": time.time()}
        elif "<offer>" in decoded_words:
            raw_example["events"][i] = {"action": "offer", "agent": 1, "data": {"price": 13000.0, "sides": ""},
                                        "time": time.time(), "start_time": time.time()}
        else:
            raw_example["events"][i] = {"action": "message", "agent": 1, "data": decoded_words, "time": time.time(),
                                        "start_time": time.time()}

        # BLEU
        #pdb.set_trace()
        sm = SmoothingFunction()
        ref_words = ref_words.split()
        ngram_weights = [0.25] * min(4, len(ref_words))
        bleu_score = sentence_bleu([ref_words], decoded_words.split(" "),
                                   weights=ngram_weights, smoothing_function=sm.method3)
        bleu_scores.append(bleu_score)

    #pdb.set_trace()
    # calculate accuracy
    d_acc = Counter()
    d_acc_total = Counter()
    utterances = parse_example(Example.from_dict(raw_example, Scenario), price_tracker, templates)[2:]
    for i in range(len(datapoint["agent_list"][1:])): # skip first start
        if datapoint['agent_list'][i+1] == 1: # if seller. also skip start
            try:
                if utterances[i].lf.to_dict()['intent'] == datapoint['dial_acts'][i+1]: # dial act +1 as it starts from 1
                    d_acc[datapoint['dial_acts'][i+1]] += 1
                d_acc_total[datapoint['dial_acts'][i+1]] += 1
            except:
                pdb.set_trace()

    # calculate my labeled data accuracy
    # wrong_labels = ["intro", "inquiry", "inform", "disagree", "agree", "vague-price"]
    # TODO  = ============
    #pdb.set_trace()
    label_acc = Counter()
    label_total = Counter()
    # labels_uuid = json.load(open("labels_uuid")) # seems to be true uuid TODO
    #
    # if uuid in labels_uuid:
    #     for i in range(len(raw_example["events"])):
    #         if input_variable_seq[i][0] == "1":
    #             # if utterances[i].lf.to_dict()["intent"] in wrong_labels:
    #             #     continue
    #             if "<" + utterances[i].lf.to_dict()["intent"] + ">" == labels_uuid[uuid][i]:
    #                 label_acc[labels_uuid[uuid][i]] += 1
    #             label_total[labels_uuid[uuid][i]] += 1
    # TOODO ================

    # extract fine acts
    fine_acts, fine_acts_binary = extract_acts(raw_example)
    fine_acts = fine_acts  # [1:]
    seq_of_fine_intent_raw = datapoint['strategies']  # [1:] TODO

    if test_eval:
        fine_acts_acc = Counter()
        fine_acts_total = Counter()
        bigram_fine_acts_acc = Counter()
        bigram_fine_acts_total = Counter()
    else:
        # Comes here
        fine_acts_acc = Counter({"a": 1})
        fine_acts_total = Counter({"a": 1})
        bigram_fine_acts_acc = Counter({"a": 1})
        bigram_fine_acts_total = Counter({"a": 1})

    # for i, _ in enumerate(seq_of_fine_intent_raw):
    #     if i + 1 >= len(fine_acts_binary):
    #         continue

    #     key_name = [0]*len(recommendation_feature_mapping)
    #     for act in range(len(seq_of_fine_intent_raw[i])-1):
    #         act = seq_of_fine_intent_raw[i][act]
    #         if act.replace("<","").replace(">","") not in recommendation_feature_mapping:
    #             continue
    #         key_name[recommendation_feature_mapping[act.replace("<","").replace(">","")]] = 1
    #     key_name = "".join([str(x) for x in key_name])

    #     if key_name not in ns_bigram_prob:
    #         continue

    # for r, r_v in enumerate(ns_bigram_prob[key_name]):
    #     if r_v > 0:
    #         if fine_acts_binary[i+1][r] == 1:
    #             fine_acts_acc[recommendation_feature_mapping_re[r]] += 1
    #         fine_acts_total[recommendation_feature_mapping_re[r]] += 1
    # for r, r_v in enumerate(ns_bigram_raw[key_name]):
    #     if r_v in fine_acts[i+1]:
    #         fine_acts_acc[r_v] += 1
    #     fine_acts_total[r_v] += 1

    # for r, r_v in enumerate(ns_bigram_raw[key_name]):
    #     for ac_i, ac in enumerate(fine_acts[i+1]):
    #         if ac == r_v:
    #             fine_acts_acc[r_v] += 1
    #             fine_acts[i+1][ac_i] = "-1"
    #             break
    #     fine_acts_total[r_v] += 1

    # unigram
    # TODO ================== Convert ACC TO F1
    for u in range(1, min(len(raw_example["events"]) + 1, len(seq_of_fine_intent_raw))):
        if raw_example["events"][u - 1]["agent"] == 0: # Current seller means.
            continue
        for a in range(len(seq_of_fine_intent_raw[u]) - 1):
            a = seq_of_fine_intent_raw[u][a]
            # for ttt in range(len(fine_acts[u])):
            #     if fine_acts[u][ttt] != '<end>':
            #         fine_acts[u][ttt] = recommendation2uniformstrategymapping[fine_acts[u][ttt].replace('<','').replace('>','')]
            if '<'+a+'>' in fine_acts[u]:
                fine_acts_acc[a] += 1
            fine_acts_total[a] += 1
    #pdb.set_trace()
    if test_eval:
        # bigram
        for u in range(1, min(len(raw_example["events"]) + 1, len(seq_of_fine_intent_raw))):
            if raw_example["events"][u - 1]["agent"] == 0:
                continue

            if u == 1:
                key = ["0"] * len(recommendation_feature_mapping)
                key = "".join(key)
                key = ""
            else:
                key = ["0"] * len(recommendation_feature_mapping)
                for sf in seq_of_fine_intent_raw[u - 1]:
                    sf = sf.replace("<", "").replace(">", "")
                    if sf in recommendation_feature_mapping:
                        key[recommendation_feature_mapping[sf]] = "1"
                key = "".join(key)
                key = "".join(seq_of_fine_intent_raw[u - 1])

            tmp_fine_acts_acc = Counter()
            tmp_fine_acts_total = Counter()
            best_acc = 0.0
            for target_bigram in ns_bigram[key]:
                tmp_tmp_fine_acts_acc = Counter()
                tmp_tmp_fine_acts_total = Counter()
                tmp_fine_acts = copy.deepcopy(fine_acts[u])
                for a in range(len(target_bigram) - 1):
                    a = target_bigram[a]
                    for faa_i, faa in enumerate(tmp_fine_acts):
                        if a == faa:
                            tmp_fine_acts[faa_i] = -1
                            tmp_tmp_fine_acts_acc[a] += 1
                            break
                    tmp_tmp_fine_acts_total[a] += 1
                if sum(tmp_tmp_fine_acts_total.values()) != 0 and 1.0 * sum(tmp_tmp_fine_acts_acc.values()) / (
                sum(tmp_tmp_fine_acts_total.values())) > best_acc:
                    tmp_fine_acts_total = tmp_tmp_fine_acts_total
                    tmp_fine_acts_acc = tmp_tmp_fine_acts_acc
                    best_acc = 1.0 * sum(tmp_tmp_fine_acts_acc.values()) / (sum(tmp_tmp_fine_acts_total.values()))

            bigram_fine_acts_acc += tmp_fine_acts_acc
            bigram_fine_acts_total += tmp_fine_acts_total

            for a in range(len(target_bigram) - 1):
                a = target_bigram[a]
                for faa_i, faa in enumerate(fine_acts[u]):
                    if a == faa:
                        fine_acts[u][faa_i] = -1
                        tmp_tmp_fine_acts_acc[a] += 1
                tmp_tmp_fine_acts_total[a] += 1
    # TODO ==========

    metrics = {}

    if n_totals == 0:
        return -1
    metrics['d_acc'] = d_acc
    metrics['d_acc_total'] = d_acc_total
    metrics['label_acc'] = label_acc
    metrics['label_acc_total'] = label_total
    metrics['fine_acts_acc'] = fine_acts_acc
    metrics['fine_acts_total'] = fine_acts_total
    metrics['bigram_fine_acts_acc'] = bigram_fine_acts_acc
    metrics['bigram_fine_acts_total'] = bigram_fine_acts_total
    metrics['bleu_score'] = bleu_scores
    metrics['TP'] = TP
    metrics['FP'] = FP
    metrics['TN'] = TN
    metrics['FN'] = FN

    return sum(print_losses) / n_totals, sum(print_losses) / u_totals, sum(print_losses), metrics, na_preds, na_labels

cluster_strategies = {"<99>": 0, "<98>": 1, "<97>": 2, "<96>": 3, "<95>": 4, "<94>": 5, "<93>": 6, "<92>": 7, "<91>": 8, "<90>": 9, "<9>": 10, "<89>": 11, "<88>": 12, "<87>": 13, "<86>": 14, "<85>": 15, "<84>": 16, "<83>": 17, "<82>": 18, "<81>": 19, "<80>": 20, "<8>": 21, "<79>": 22, "<78>": 23, "<77>": 24, "<76>": 25, "<75>": 26, "<74>": 27, "<73>": 28, "<72>": 29, "<71>": 30, "<70>": 31, "<7>": 32, "<69>": 33, "<68>": 34, "<67>": 35, "<66>": 36, "<65>": 37, "<64>": 38, "<63>": 39, "<62>": 40, "<61>": 41, "<60>": 42, "<6>": 43, "<59>": 44, "<58>": 45, "<57>": 46, "<56>": 47, "<55>": 48, "<54>": 49, "<53>": 50, "<52>": 51, "<51>": 52, "<50>": 53, "<5>": 54, "<49>": 55, "<48>": 56, "<47>": 57, "<46>": 58, "<45>": 59, "<44>": 60, "<43>": 61, "<42>": 62, "<41>": 63, "<40>": 64, "<4>": 65, "<39>": 66, "<38>": 67, "<37>": 68, "<36>": 69, "<35>": 70, "<34>": 71, "<33>": 72, "<32>": 73, "<31>": 74, "<30>": 75, "<3>": 76, "<299>": 77, "<298>": 78, "<297>": 79, "<296>": 80, "<295>": 81, "<294>": 82, "<293>": 83, "<292>": 84, "<291>": 85, "<290>": 86, "<29>": 87, "<289>": 88, "<288>": 89, "<287>": 90, "<286>": 91, "<285>": 92, "<284>": 93, "<283>": 94, "<282>": 95, "<281>": 96, "<280>": 97, "<28>": 98, "<279>": 99, "<278>": 100, "<277>": 101, "<276>": 102, "<275>": 103, "<274>": 104, "<273>": 105, "<272>": 106, "<271>": 107, "<270>": 108, "<27>": 109, "<269>": 110, "<268>": 111, "<267>": 112, "<266>": 113, "<265>": 114, "<264>": 115, "<263>": 116, "<262>": 117, "<261>": 118, "<260>": 119, "<26>": 120, "<259>": 121, "<258>": 122, "<257>": 123, "<256>": 124, "<255>": 125, "<254>": 126, "<253>": 127, "<252>": 128, "<251>": 129, "<250>": 130, "<25>": 131, "<249>": 132, "<248>": 133, "<247>": 134, "<246>": 135, "<245>": 136, "<244>": 137, "<243>": 138, "<242>": 139, "<241>": 140, "<240>": 141, "<24>": 142, "<239>": 143, "<238>": 144, "<237>": 145, "<236>": 146, "<235>": 147, "<234>": 148, "<233>": 149, "<232>": 150, "<231>": 151, "<230>": 152, "<23>": 153, "<229>": 154, "<228>": 155, "<227>": 156, "<226>": 157, "<225>": 158, "<224>": 159, "<223>": 160, "<222>": 161, "<221>": 162, "<220>": 163, "<22>": 164, "<219>": 165, "<218>": 166, "<217>": 167, "<216>": 168, "<215>": 169, "<214>": 170, "<213>": 171, "<212>": 172, "<211>": 173, "<210>": 174, "<21>": 175, "<209>": 176, "<208>": 177, "<207>": 178, "<206>": 179, "<205>": 180, "<204>": 181, "<203>": 182, "<202>": 183, "<201>": 184, "<200>": 185, "<20>": 186, "<2>": 187, "<199>": 188, "<198>": 189, "<197>": 190, "<196>": 191, "<195>": 192, "<194>": 193, "<193>": 194, "<192>": 195, "<191>": 196, "<190>": 197, "<19>": 198, "<189>": 199, "<188>": 200, "<187>": 201, "<186>": 202, "<185>": 203, "<184>": 204, "<183>": 205, "<182>": 206, "<181>": 207, "<180>": 208, "<18>": 209, "<179>": 210, "<178>": 211, "<177>": 212, "<176>": 213, "<175>": 214, "<174>": 215, "<173>": 216, "<172>": 217, "<171>": 218, "<170>": 219, "<17>": 220, "<169>": 221, "<168>": 222, "<167>": 223, "<166>": 224, "<165>": 225, "<164>": 226, "<163>": 227, "<162>": 228, "<161>": 229, "<160>": 230, "<16>": 231, "<159>": 232, "<158>": 233, "<157>": 234, "<156>": 235, "<155>": 236, "<154>": 237, "<153>": 238, "<152>": 239, "<151>": 240, "<150>": 241, "<15>": 242, "<149>": 243, "<148>": 244, "<147>": 245, "<146>": 246, "<145>": 247, "<144>": 248, "<143>": 249, "<142>": 250, "<141>": 251, "<140>": 252, "<14>": 253, "<139>": 254, "<138>": 255, "<137>": 256, "<136>": 257, "<135>": 258, "<134>": 259, "<133>": 260, "<132>": 261, "<131>": 262, "<130>": 263, "<13>": 264, "<129>": 265, "<128>": 266, "<127>": 267, "<126>": 268, "<125>": 269, "<124>": 270, "<123>": 271, "<122>": 272, "<121>": 273, "<120>": 274, "<12>": 275, "<119>": 276, "<118>": 277, "<117>": 278, "<116>": 279, "<115>": 280, "<114>": 281, "<113>": 282, "<112>": 283, "<111>": 284, "<110>": 285, "<11>": 286, "<109>": 287, "<108>": 288, "<107>": 289, "<106>": 290, "<105>": 291, "<104>": 292, "<103>": 293, "<102>": 294, "<101>": 295, "<100>": 296, "<10>": 297, "<1>": 298, "<0>": 299}
def up_weight_cluster(strategies):
    # cluster_info = json.load(open("cluster_info"))
    # #preprocessing
    # for key in cluster_info:
    #     result_element = cluster_info[key][0]
    #     for element in cluster_info[key]:
    #         result_element = list(set(result_element) & set(element))
    #     cluster_info[key] = result_element
    
    # with open("cluster_info_preprocessed", "w") as output:
    #     json.dump(cluster_info, output)

    cluster_info = json.load(open("cluster_info_preprocessed"))
    up_weight_clusters = []
    for key in cluster_info:
        for strat in strategies:
            if strat in cluster_info[key] and len(cluster_info[key]) < 2:
                up_weight_clusters.append(key)
    return up_weight_clusters

def generate_tokens(searcher, sentence, scenario, scene_index):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = word_to_index_eval(sentence)
    seq_of_dialogs.append(np.reshape(indexes_batch, (-1, batch_size)))

    # Transpose dimensions of batch to match models' expectations
    # Decode sentence with searcher
    test_state_embeddings.append(wfsm.look_up_state_embedding(wfsm.current_state))
    test_state_embeddings_bs.append(wfsm_bs.look_up_state_embedding(wfsm_bs.current_state))

    # if len(test_state_embeddings) > 3:
    #     test_state_embeddings[-1][8] = 3.5
    # up_weight_clusters = up_weight_cluster(["<liwc_informal>"])
    # for clu in up_weight_clusters:
    #     test_state_embeddings_bs[-1][cluster_strategies[clu]] = 3.0


    tokens, scores = searcher(seq_of_dialogs, test_state_embeddings, test_state_embeddings_bs, scene_index, with_state_embedding, bs_flag)
    tokens = [token.item() for token in tokens]
    seq_of_dialogs.append(np.reshape(tokens, (-1, batch_size)))

    # indexes -> words
    decoded_words =  index_to_word(tokens, scenario)
    return decoded_words

def evaluateInput(searcher):
    #initilization
    global test_state_embeddings
    global test_state_embeddings_bs
    global seq_of_dialogs

    seq_of_dialogs = [[[0]]]
    test_state_embeddings = list()
    test_state_embeddings_bs = list()
    wfsm.current_state = 0
    wfsm_bs.current_state = 0
    
    scenario_path = "scenario.json"
    selected_scenario = json.load(open("selected_scenario"))
    # raw_scenarios = json.load(open(scenario_path))
    # for s in raw_scenarios:
    #     if s["uuid"] == "S_v5aN2VMjBToGVuQ8":
    #         selected_scenario = s
    #         break

    print ("Scenario:")
    print (selected_scenario["kbs"][1]["item"]["Title"] + "\n" + " ".join(selected_scenario["kbs"][1]["item"]["Description"]))

    input_sentence = ''
    raw_example = {
    "uuid":"C_af8b847888704a0d91b3ad30393c0907",
    "scenario": selected_scenario,
    "agents":{  
      "1":"human",
      "0":"human"
      },
    "scenario_uuid":selected_scenario["uuid"],
    "events":[],
    "outcome": {"reward": 1, "offer": {"price": 13000.0, "sides": ""}}
    }
    wfsm.step("0:<start>")
    test_state_embeddings.append(wfsm.look_up_state_embedding(wfsm.current_state))
    test_state_embeddings_bs.append(wfsm_bs.look_up_state_embedding(wfsm_bs.current_state))

    scene = (raw_example["scenario"]["kbs"][1]["item"]["Category"] + " " + " ".join(raw_example["scenario"]["kbs"][1]["item"]["Description"])+ " " + raw_example["scenario"]["kbs"][1]["item"]["Title"])
    scene = normalizeString(scene, {"price":None}, raw_example["scenario"]) 

    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
        except KeyError:
            print("Error: Encountered unknown word.")
        if "<accept>" in input_sentence:
            raw_example["events"].append({
                "action":"accept",
                "agent": 0,
                "time":time.time(),
                "start_time":time.time()
                })
            break
        elif "<reject>" in input_sentence:
            raw_example["events"].append({
                "action":"reject",
                "agent": 0,
                "time":time.time(),
                "start_time":time.time()
                })
            break
        elif "<quit>" in input_sentence:
            raw_example["events"].append({
                "action":"quit",
                "agent": 0,
                "time":time.time(),
                "start_time":time.time()
                })
            break
        elif "<offer>" in input_sentence:
            raw_example["events"].append({
                "action":"offer",
                "agent": 0,
                "data":{  
                    "price":13000.0,
                    "sides":""
                 },
                 "time":time.time(),
                "start_time":time.time()
                })
        else:
            raw_example["events"].append({
                "action":"message",
                "agent": 0,
                "data": input_sentence,
                "time":time.time(),
                "start_time":time.time()
                })
        # Evaluate sentence
        # Normalize sentence
        utterance = parse_example(Example.from_dict(raw_example, Scenario), price_tracker, templates)[-1]
        tmp_dict = utterance.lf.to_dict()
        wfsm.step("0:" + tmp_dict["intent"])

        
        _, l_tmp = extract_acts(raw_example)
        l_tmp = l_tmp[-1]
        wfsm_bs.step("<" + str(kmeans.predict([np.multiply(coef_weights,l_tmp)])[0]) + ">")

        input_sentence = normalizeString(input_sentence, tmp_dict, raw_example["scenario"])
        output_words = generate_tokens(searcher, input_sentence, raw_example["scenario"], word_to_index_eval(scene))
        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == '<eos>')]
        decoded_words = ' '.join(output_words)
        print('Bot:', decoded_words)

        if "<accept>" in decoded_words:
            raw_example["events"].append({
                "action":"accept",
                "agent": 1,
                "time":time.time(),
                "start_time":time.time()
                }) 
            break
        elif "<reject>" in decoded_words:
            raw_example["events"].append({
                "action":"reject",
                "agent": 1,
                "time":time.time(),
                "start_time":time.time()
                })
            break
        elif "<quit>" in decoded_words:
            raw_example["events"].append({
                "action":"quit",
                "agent": 1,
                "time":time.time(),
                "start_time":time.time()
                })
            break
        elif "<offer>" in decoded_words:
            raw_example["events"].append({
                "action":"offer",
                "agent": 1,
                "data":{  
                    "price":13000.0,
                    "sides":""
                 },
                 "time":time.time(),
                "start_time":time.time()
                })
        else:
            raw_example["events"].append({
                "action":"message",
                "agent": 1,
                "data": decoded_words,
                "time":time.time(),
                "start_time":time.time()
                })

        #parse first and use wfsm to move forward
        utterance = parse_example(Example.from_dict(raw_example, Scenario), price_tracker, templates)[-1]
        tmp_dict = utterance.lf.to_dict()
        wfsm.step("1:" + tmp_dict["intent"])
        test_state_embeddings.append(wfsm.look_up_state_embedding(wfsm.current_state))

        #parse dialog acts
        _, l_tmp = extract_acts(raw_example)
        l_tmp = l_tmp[-1]
        wfsm_bs.step("<" + str(kmeans.predict([np.multiply(coef_weights, l_tmp)])[0]) + ">")
        test_state_embeddings_bs.append(wfsm_bs.look_up_state_embedding(wfsm_bs.current_state))


    ##########################
    #analyise state embedding#
    ##########################
    #check whether the embedding is giving any useful info
    for embed in test_state_embeddings:
        print ([(x,y) for y,x in sorted(zip(embed,dialog_acts), reverse=True)][:5])

    for embed in test_state_embeddings_bs:
        print ([(x,y) for y,x in sorted(zip(embed,cluster_strategies.keys()), reverse=True)][:5])

def main():
    #set random seed for reprocducibility
    random_seed = 2
    if random_seed != -1:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

    #read raw data
    data = pickle.load(open(utils.PROJ_DIR + 'data/negotiation_data/data/strategy_vector_data_FULL_Yiheng.pkl', 'rb'))
    negotiation_lbl2id = data['strategies2colid']
    da_lbl2id          = data['dialacts2id']

    vocab['word2id']   = data['word2id']
    vocab['id2word']   = {v:k for k,v in vocab['word2id'].items()}
    vocab['vocab_size']= len(vocab['word2id'])

    # set feature weights. Not for "start"
    strat_freq = ddict(int)
    da_labels = []
    tot_labels = 0
    for dat in data['train']:
        for strat in dat['strategies_vec']:
            for idx in np.where(np.array(strat) == 1)[0]:
                if idx != negotiation_lbl2id['<start>']:
                    strat_freq[idx] += 1
        for da in dat['dial_acts_vec']:
            if da != da_lbl2id['<start>']:
                da_labels.append(da)
    tot_labels = len(da_labels)  # these will be without 0
    label_set = np.unique(da_labels)
    strat_feature_weights = {k: ((tot_labels - v) / v) for k, v in strat_freq.items()}
    from sklearn.utils.class_weight import compute_class_weight
    da_weights = compute_class_weight('balanced', label_set, da_labels)
    da_feature_weights = {label_set[idx]: da_weights[idx] for idx in range(len(label_set))}

    num_strat = len(negotiation_lbl2id) - 1 # Not start
    num_da = len(da_lbl2id) - 1 # Not start
    print(num_strat)
    print(num_da)

    loadFilename = None
    if not train_flag:
        loadFilename = os.path.join(save_dir, model_name, corpus_name,
                                  '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                                  '{}_checkpoint.tar'.format(checkpoint_iter))
    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename, map_location=device)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        sentence_sd = checkpoint['se']
        d_classifier_sd = checkpoint["dc"]
        na_classifier_sd = checkpoint["na"]
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        sen_encoder_optimizer_sd = checkpoint['se_opt']
        d_classifier_optimizer_sd = checkpoint['dc_opt']
        na_classifier_optimizer_sd = checkpoint['na_opt']
        if true_joint:
            universal_optimizer_sd = checkpoint['uni_opt']
        embedding_sd = checkpoint['embedding']
        if use_cluster_bs:
            cluster_embeddings_sd = checkpoint['cluster_embeddings']
        strat_encoder_optimizer_sd = checkpoint['strat_encoder_opt']
        da_encoder_optimizer_sd = checkpoint['da_encoder_opt']


    print('Building encoder and decoder ...')
    # Initialize word embeddings
    import gensim
    #pretrain word2vec
    if pre_train:
        print ("Loading word2vec word embedding...")
        pretrained_word_embedding = list()
        w_model = gensim.models.KeyedVectors.load_word2vec_format("/projects/tir1/users/yihengz1/GoogleNews-vectors-negative300.bin",binary=True)
        for word in vocab['word2id']:
          if word not in w_model.wv:
            pretrained_word_embedding.append(np.random.rand(hidden_size,))
          else:
            pretrained_word_embedding.append(w_model.wv[word].tolist())
        embedding = nn.Embedding(vocab['vocab_size'], hidden_size)
        embedding.weight = nn.Parameter(torch.FloatTensor(pretrained_word_embedding).to(device))
        embedding.weight.requires_grad = False
    else:
        embedding = nn.Embedding(vocab['vocab_size'], hidden_size)

    # cluster_embeddings = nn.Embedding(int(bag_of_strategies_filename.split("_")[-1]) + 1, int(bag_of_strategies_filename.split("_")[-1])) #TODO Not to be used
    # if loadFilename:
    #     embedding.load_state_dict(embedding_sd)
    #     if use_cluster_bs:
    #         cluster_embeddings.load_state_dict(cluster_embeddings_sd)
    
    embedding.to(device)
    # cluster_embeddings.to(device)
    print ("Loading Embedding Complete.")
    
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN2(attn_model, embedding, hidden_size, vocab['vocab_size'], num_da, num_strat, decoder_n_layers, dropout, dc_input)
    sen_encoder = SentenceEncoder(hidden_size, embedding, sen_encoder_n_layers, dropout)
    d_classifier = Dialog_Act_Classifier(hidden_size, num_da)
    na_classifier = NA_Classifier(hidden_size, num_strat)
    # strat_encoder = StrategyEncoder(hidden_size, num_strat)
    # da_encoder    = StrategyEncoder(hidden_size, num_da)
    # strat_encoder = make_model(None, None, num_strat, 6, num_strat)
    # da_encoder = make_model(None, None, num_da, 6, num_da)
    strat_encoder = WFSTModel('/projects/tir1/users/rjoshi2/negotiation/negotiation_personality/data/negotiation_data/data/seq_end_strats_rjyiheng_train_rjyiheng.wfst', negotiation_lbl2id)
    da_encoder = WFSTModelDA('/projects/tir1/users/rjoshi2/negotiation/negotiation_personality/data/negotiation_data/data/seq_da_acts_rjyiheng_train_rjyiheng.wfst', da_lbl2id)

    models = {
        'utt_encoder': encoder,
        'utt_decoder': decoder,
        'dial_encoder': sen_encoder,
        'da_classifier': d_classifier,
        'strat_classifier': na_classifier,
        'strat_encoder': strat_encoder,
        'da_encoder': da_encoder
    }
    
    if loadFilename:
        models['utt_encoder'].load_state_dict(encoder_sd)
        models['utt_decoder'].load_state_dict(decoder_sd)
        models['dial_encoder'].load_state_dict(sentence_sd)
        models['da_classifier'].load_state_dict(d_classifier_sd)
        models['strat_classifier'].load_state_dict(na_classifier_sd)
        models['da_encoder'].load_state_dict(da_encoder)
        models['strat_encoder'].load_state_dict(strat_encoder)
    # Use appropriate device
    for model in models:
        models[model].to(device)
    print('Models built and ready to go!')

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(models['utt_encoder'].parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(models['utt_decoder'].parameters(), lr=learning_rate * decoder_learning_ratio)
    sen_encoder_optimizer = optim.Adam(models['dial_encoder'].parameters(), lr=learning_rate)
    d_classifier_optimizer = optim.Adam(models['da_classifier'].parameters(), lr=learning_rate)
    na_classifier_optimizer = optim.Adam(models['strat_classifier'].parameters(), lr=learning_rate)
    strat_encoder_optimizer = optim.Adam(models['strat_encoder'].parameters(), lr=learning_rate)
    da_encoder_optimizer = optim.Adam(models['da_encoder'].parameters(), lr=learning_rate)
    if true_joint:
        universal_optimizer = optim.Adam(list(models['dial_encoder'].parameters()) + \
                                         list(models['utt_decoder'].parameters()) + \
                                         list(models['utt_encoder'].parameters()) + \
                                         list(models['da_classifier'].parameters()) + \
                                         list(models['strat_classifier'].parameters()) + \
                                         list(models['strat_encoder'].parameters()) + \
                                         list(models['da_encoder'].parameters()), lr=learning_rate)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)
        sen_encoder_optimizer.load_state_dict(sen_encoder_optimizer_sd)
        d_classifier_optimizer.load_state_dict(d_classifier_optimizer_sd)
        na_classifier_optimizer.load_state_dict(na_classifier_optimizer_sd)
        if true_joint:
            universal_optimizer.load_state_dict(universal_optimizer_sd)
        strat_encoder_optimizer.load_state_dict(strat_encoder_optimizer_sd)
        da_encoder_optimizer.load_state_dict(da_encoder_optimizer_sd)
    optimizers = {
        'utt_encoder': encoder_optimizer,
        'utt_decoder': decoder_optimizer,
        'dial_encoder': sen_encoder_optimizer,
        'da_classifier': d_classifier_optimizer,
        'strat_classifier': na_classifier_optimizer,
        'strat_encoder': strat_encoder_optimizer,
        'da_encoder': da_encoder_optimizer,
    }
    if true_joint:
        optimizers['universal'] = universal_optimizer

    criteria = nn.CrossEntropyLoss()

    # #raw_bag_of_strategies
    # raw_bag_of_strategies = json.load(open("bag_of_strategies"))
    params = {
        'encoder_n_layers': encoder_n_layers,
        'decoder_n_layers': decoder_n_layers,
        'save_dir': save_dir,
        'n_iteration': n_iteration,
        'batch_size': batch_size,
        'print_every': print_every,
        'save_every': save_every,
        'clip': clip,
        'corpus_name': corpus_name,
        'loadFilename': loadFilename,
        'model_name': model_name,
        'hidden_size': hidden_size,
        'num_da': num_da,
        'num_strat': num_strat
    }
    #Run training iterations
    if train_flag:
        # Ensure dropout layers are in train mode
        for model in models:
            models[model].train()
        print("Starting Training!")
        if joint:
            if true_joint:
                true_joint_trainIters(cluster_embeddings, raw_bag_of_strategies, seq_of_fine_intents_raw, seq_of_fine_intents, scenario_index, model_name, input_dialogs, seq_of_intents_train, dev_scenario_index, dev_dialogs, seq_of_intents_dev, models['utt_encoder'], d_classifier, decoder, sen_encoder, encoder_optimizer, d_classifier_optimizer ,decoder_optimizer, sen_encoder_optimizer,
                    embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                    print_every, save_every, clip, corpus_name, loadFilename, criteria, na_classifier, na_classifier_optimizer, universal_optimizer)
            else:
                # TODO HERE!!!
                joint_trainIters(data['train'], data['valid'], models, optimizers, embedding, criteria, params)
        else:
            trainIters(cluster_embeddings, raw_bag_of_strategies, seq_of_fine_intents_raw, seq_of_fine_intents, scenario_index, model_name, input_dialogs, seq_of_intents_train, dev_scenario_index, dev_dialogs, seq_of_intents_dev, encoder, d_classifier, decoder, sen_encoder, encoder_optimizer, d_classifier_optimizer ,decoder_optimizer, sen_encoder_optimizer,
                embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                print_every, save_every, clip, corpus_name, loadFilename, criteria, na_classifier, na_classifier_optimizer)
    
    #evaluation and interactive interface
    else:
        for model in models:
            models[model].eval()

        #evaluate on dev set
        print ("==================Evaluate on dev set================")
        evaluateIters_rj(data['valid'], models, optimizers, embedding, criteria, params, test_eval=False) # originally True
        print("==================Evaluate on test set================")
        evaluateIters_rj(data['test'], models, optimizers, embedding, criteria, params, test_eval=False)
        pdb.set_trace()
        #initialize searcher
        searcher = GreedySearchDecoder(models, embedding, True)
        
        while True:
            evaluateInput(searcher)
            try:
                # Get input sentence
                input_sentence = input('wanna do it again?(y/n)')
            except KeyError:
                print("Error: Encountered unknown word.")
            if input_sentence == "y":
                continue
            else:
                break

if __name__ == "__main__": main()
