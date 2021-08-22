from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict as ddict
import matplotlib.pyplot as plt
import numpy as np
import pdb, unicodedata
import os, re

#PROJ_DIR = '/usr1/home/rjoshi2/negotiation_personality/'

curr_file_path = os.path.dirname(os.path.abspath(__file__)) + '/'
PROJ_DIR = os.path.dirname(os.path.dirname(curr_file_path)) + '/'

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



def print_conv(data):
    '''
    Takes data (eg train_data[1210])
    Prints conversation nicely
    '''
    print ('\n'.join(['AGENT '+str(i['agent']) + ' : ' + i['data'] for i in data['events'] if i['action'] not in ['quit', 'accept', 'offer', 'reject']]))

def get_conv(data, split = False):
    '''
    Takes data (eg train_data[1210])
    if split is true then :
        Returns conversation as a list of tuple (agent, sent, before/after) 
    else:
        Returns conversation as a list of tuple (agent, sent)
    Agent is either buyer (0) or seller (1)
    before after is 0 or 1. 0 signifies before buyer_propose point
    after signifies after_propose point
    '''
    conv = []
    if not split:
        for i in data['events']:
            if i['action'] not in ['quit', 'accept', 'offer', 'reject']:
                conv.append((i['agent'], i['data']))
        #return [(str(i['agent']), i['data'] for i in data['events'] if i['action'] not in ['quit', 'accept', 'offer', 'reject'])]
        return conv
    else:
        if 'strategies' not in data['events'][0]:
            print ('ERROR : No Strategies in data. Did you mean to put split as false? If not, then provide data with strategies so that we can find buyer propose.')
            return []
        buyer_propose_idx = -1
        for j in range(len(data['events'])):
            if type(data['events'][j]['data']) != str: #Not interested in action symbols (conclude etc)
                continue
            # get if buyer_propose
            if buyer_propose_idx == -1 and data['events'][j]['agent'] == 0:
                for strategy in data['events'][j]['strategies']:
                    if strategy == 'buyer_propose':
                        buyer_propose_idx = j
                        break
                
            # Add conversations
            if buyer_propose_idx == -1:
                conv.append((data['events'][j]['agent'], data['events'][j]['data'], 0))
            else:
                conv.append((data['events'][j]['agent'], data['events'][j]['data'], 1))
        return conv
                    
    
def getRatio(datapoint):
    '''Takes datapoint and returns ratio. Return None if datapoint not accepted'''
    buyer_amt = datapoint['outcome']['buyer_start_amt']
    seller_amt = datapoint['outcome']['seller_suggested_amt']
    final_amt = datapoint['outcome']['accepted_amt']
    if final_amt == -1:
        return -99999
    if buyer_amt == -1:
        if datapoint['scenario']['kbs'][0]['personal']['Role'] == 'buyer':
            buyer_amt = datapoint['scenario']['kbs'][0]['personal']['Target']
        else:
            buyer_amt = datapoint['scenario']['kbs'][1]['personal']['Target']
    if seller_amt == -1:
        if datapoint['scenario']['kbs'][0]['personal']['Role'] == 'seller':
            seller_amt = datapoint['scenario']['kbs'][0]['personal']['Target']
        else:
            seller_amt = datapoint['scenario']['kbs'][1]['personal']['Target']
    curr_norm = (final_amt - buyer_amt) / (seller_amt - buyer_amt)
    return curr_norm

def getAmounts(datapoint):
    '''
    Takes a data point and returns buyer_amt, seller_amt, final_amt and is_accept
    Can update it to include the *first_proposed* price.. to see better how things changed
    '''
#     is_accept = False
#     buyer_amt, seller_amt, final_amt = 0, 0, 0
#     for j in range(len(datapoint['scenario']['kbs'])):
#         if datapoint['scenario']['kbs'][j]['personal']['Role'] == 'buyer': 
#             buyer_amt = datapoint['scenario']['kbs'][j]['personal']['Target']
#         if datapoint['scenario']['kbs'][j]['personal']['Role'] == 'seller': 
#             seller_amt = datapoint['scenario']['kbs'][j]['personal']['Target']
#     for j in range(len(datapoint['events'])):
#         if datapoint['events'][j]['action'] == 'offer':
#             final_amt = datapoint['events'][j]['data']['price']
#         if datapoint['events'][j]['action'] == 'accept': is_accept = True
    buyer_amt = datapoint['outcome']['buyer_start_amt']
    seller_amt = datapoint['outcome']['seller_suggested_amt']
    final_amt = datapoint['outcome']['accepted_amt']
    ### NEW ADDITION
    if buyer_amt == -1:
        if datapoint['scenario']['kbs'][0]['personal']['Role'] == 'buyer':
            buyer_amt = datapoint['scenario']['kbs'][0]['personal']['Target']
        else:
            buyer_amt = datapoint['scenario']['kbs'][1]['personal']['Target']
    if seller_amt == -1:
        if datapoint['scenario']['kbs'][0]['personal']['Role'] == 'seller':
            seller_amt = datapoint['scenario']['kbs'][0]['personal']['Target']
        else:
            seller_amt = datapoint['scenario']['kbs'][1]['personal']['Target']
    ### NEW ADDITION
    if final_amt == -1:
        is_accept = False
    else:
        is_accept = True
    return buyer_amt, seller_amt, final_amt, is_accept

def get_strategies(data_point):
    '''
    Takes data point and returns list of strategies
    '''
    return [(x['agent'], x['data'], x['strategies']) for x in data_point['events'] if type(x['data']) == str]

def toTake(datapoint):
    '''
    Takes a datapoint and decides if it should be taken or not
    '''
    conv = get_conv(datapoint)
    if len(conv) < 5: 
        return False
    ratio = getRatio(datapoint)
    if ratio == -99999:
        return False
    if ratio < -10 or ratio > 10:
        return False
    return True

def getContext(sent, word, window_size = 5):
    '''
    Takes a sent and word, and returns the context of it as list of string
    If multiple instances, return a list of list of string.
    '''
    if word not in sent:
        return []
#     import string
    wrdlist = word_tokenize(sent)
    indices = [i for i, x in enumerate(wrdlist) if x == word]
    context = []
    for idx in indices:
#         if idx == -1:
#             print ("Some error : ", word, " : ", sent)
#             return []
    #     rightcontext = []
    #     i = idx
    #     while i < len(sent) and i < idx + window_size:
    #         rightcontext.append(wrdlist[i])
    #         if wrdlist[i] in string.punctuation:
    #             break

    #     i = max(i-5, 0)
    #     while i < idx:
    #         context.append(wrdlist[i])
    #     context.append(wrdlist[idx])
        context.append(wrdlist[idx-window_size:idx+window_size+1])
    return context

   

def plotOdds(mat1, mat2, id2category, title, significant_categories):
    '''
    Takes two matrices and computes the odds figure of the means
    significant_categories is list of categories to take
    And plots it
    '''
#     pdb.set_trace()
    arr2 = np.mean(mat1, axis = 0)[:] # Swapped so that analysis easier.
    arr1 = np.mean(mat2, axis = 0)[:]
    odds = arr1 / arr2
    for i in range(len(odds)):
        if odds[i] < 1:
            odds[i] = -1.0 / odds[i]
    categories = [v for k,v in id2category.items() if k in significant_categories]
    y_pos = np.arange(len(categories))
    
    fig, ax = plt.subplots()
    ax.barh(y_pos, odds, align='center')
    
    ax.set_xlim([-3, 3])
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
    ax.axvline(0, color='grey', alpha=0.25)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.invert_yaxis() # labels read top-to-bottom
    ax.set_xlabel('Odds')
    ax.set_title(title + ' Odds Ratio')
    
    plt.show()



def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s, tmp_dict, scenario, normalize_price = True):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)         # Add space around punctuation RJ
    
    #maybe i need to keep the number
    if normalize_price:
        if not tmp_dict["price"]:                   # Price given for scene None RJ IDK WHY
            s = re.sub(r"[^a-zA-Z.!?<>]+", r" ", s)
        else:
            s = bin_price(s, tmp_dict["price"], scenario)  # Price is subbed 
    else:
        s = re.sub(r"[^a-zA-Z.!?<>0-9]+", r" ", s)
    
    s = re.sub(r"\s+", r" ", s).strip()         # multiple spaces become 1
    return s

def bin_price(s, current_price, scenario):
    if scenario["kbs"][0]["personal"]["Role"] == "buyer":
        target = scenario["kbs"][0]["personal"]["Target"]
        price = scenario["kbs"][1]["personal"]["Target"]
    elif scenario["kbs"][0]["personal"]["Role"] == "seller":
        target = scenario["kbs"][1]["personal"]["Target"]
        price = scenario["kbs"][0]["personal"]["Target"]
    else:
        print ("Role is not matched!")
        exit()
    
    nor_price = 1.0 * (current_price - target)/(price - target)
    if nor_price > 2.0:
        nor_price = "<price>_2.0"
    elif nor_price < -2.0:
        nor_price = "<price>_-2.0"
    else:
        nor_price = "<price>_"+str(round(nor_price,1))

    s = re.sub(r"[^a-zA-Z.!?<>0-9]+", r" ", s)

    return re.sub(r"\d+,?\d+", " " + nor_price + " ", s)