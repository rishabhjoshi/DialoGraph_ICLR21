import json
from operator import add
from operator import sub
from sklearn.model_selection import cross_val_score
import random
import numpy as np
#import liwc_result_parser
import re
from sklearn.svm import LinearSVC
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#import sentiment_from_liu
import read_dominance_arousal_valence
from nltk.util import ngrams
from nltk import pos_tag
import LIWC_Mapping
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier
from scipy.stats.stats import pearsonr
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import sys
#sys.path.insert(0, '/projects/tir1/users/yihengz1/negotiation/evaluation/auto_labeling/multeval-0.5.1/')
#from calculate import feature_extractor
#from convert_sentence_to_parse_tree import string_to_phrases
import os
import sys
#from sklearn.externals import joblib
from collections import Counter
from importlib import reload
reload(sys)
import os
curr_file_path = os.path.dirname(os.path.abspath(__file__)) + '/'
#sys.setdefaultencoding('utf8')

lemmatizer = WordNetLemmatizer()
stopWords = stopwords.words('english')
feature_size = 47
recommendation_template = dict()

pos, neg = LIWC_Mapping.sentiment()
liwc_personal_concern = LIWC_Mapping.personal_concern()
liwc_family = LIWC_Mapping.family()
liwc_friend = LIWC_Mapping.friend()
liwc_i = LIWC_Mapping.i()
liwc_informal_dic = LIWC_Mapping.informal()
liwc_certain = LIWC_Mapping.certain()
lexicon_diff_dic_pos = ["years", "shape", "including", "yes", "apartment", "we've", "had", "good", "they", "got", "always", "works", "antique", "right", "some", "bluetooth", "tires", "are", "out", "even", "everything", "new", "we", "recently", "screen", "never", "free", "put", "months", "color", "quality", "from", "would", "it's", "there", "two", "been", "few", "too", "was", "selling", "that", "brand", "sound", "this", "car", "up", "need", "any", "i've", "amazing", "you", "nice", "used", "kept", "clean", "time"]
lexicon_diff_dic_neg = ["less", "excellent", "actually", "condition", "like", "tear", "miles", "year", "does", "newly", "come", "about", "many", "comes", "warranty", "features", "table", "your", "unit", "use", "area", "wood", "lot", "but", "scratches", "solid", "will", "piece", "almost", "as", "normal", "light", "well", "so", "original"]

hedges_word = {'claim': 11, 'presumably': 67, 'unclear': 60, 'often': 34, 'indicated': 26, 'feel': 20, 'seems': 48, 'mainly': 29, 'doubtful': 16, 'plausible': 37, 'argued': 74, 'likely': 28, 'unlikely': 62, "couldn't": 14, 'claimed': 12, 'estimated': 19, 'apparently': 3, 'supposes': 79, 'appeared': 5, 'relatively': 46, 'postulates': 81, 'guess': 24, 'appear': 4, 'would': 64, 'indicate': 25, 'perhaps': 36, 'assumed': 10, 'generally': 23, 'approximately': 7, 'should': 49, 'argues': 73, 'almost': 1, 'doubt': 15, 'suspect': 55, 'presumable': 43, 'indicates': 77, 'postulated': 42, 'probably': 45, 'postulate': 41, 'might': 32, 'ought': 35, 'supposed': 78, 'fairly': 69, 'apparent': 2, 'around': 8, 'mostly': 33, 'may': 30, 'plausibly': 38, 'felt': 21, 'essentially': 17, 'possible': 39, 'unclearly': 61, 'possibly': 40, 'feels': 76, 'somewhat': 51, 'frequently': 22, 'estimate': 18, 'quite': 70, 'appears': 6, 'probable': 44, 'suggested': 53, 'about': 0, 'uncertain': 58, 'suspects': 80, 'largely': 27, 'assume': 9, 'maybe': 31, 'could': 13, 'sometimes': 50, 'rather': 71, 'roughly': 47, 'suggests': 68, 'tended to': 66, 'uncertainly': 59, 'suppose': 54, 'broadly': 65, 'suggest': 52, 'usually': 63, 'claims': 75, 'argue': 72, 'typically': 57, 'typical': 56}
hedge_LIWC_word = {'think': 0, 'guesses': 28, 'consider': 6, 'basically': 73, 'understand': 12, 'generally': 57, 'estimates': 31, 'somehow': 77, 'speculates': 34, 'somebody': 83, 'understood': 14, 'likely': 45, 'guessed': 29, 'unlikely': 50, 'read': 53, 'speculated': 35, 'says': 55, 'seem': 21, 'estimated': 32, 'usually': 64, 'thinks': 1, 'seemed': 23, 'guess': 27, 'speculate': 33, 'appear': 18, 'suggests': 37, 'perhaps': 47, 'assumed': 11, 'apparently': 71, 'seems': 22, 'approximately': 74, 'find': 15, 'rarely': 59, 'appeared': 20, 'occasionally': 62, 'surely': 43, 'about': 70, 'probably': 44, 'several': 66, 'might': 42, 'something': 81, 'assumes': 10, 'virtually': 72, 'partially': 78, 'almost': 68, 'actually': 79, 'unsure': 48, 'somewhere': 84, 'may': 39, 'some': 67, 'supposes': 25, 'possible': 52, 'often': 58, 'possibly': 51, 'considers': 7, 'seldom': 63, 'somewhat': 76, 'practically': 69, 'frequently': 61, 'estimate': 30, 'believe': 3, 'appears': 19, 'probable': 49, 'suggested': 38, 'understands': 13, 'like': 80, 'largely': 56, 'considered': 8, 'should': 41, 'could': 40, 'sometimes': 60, 'say': 54, 'believes': 5, 'thought': 2, 'assume': 9, 'someone': 82, 'suppose': 24, 'supposed': 26, 'suggest': 36, 'believed': 4, 'found': 16, 'roughly': 75, 'finds': 17, 'maybe': 46, 'most': 65}
hedges_phrase = ["certain amount","certain extent","certain level", "from our perspective", "in general","in most cases","in most instances", "in our view", "on the whole", "from this perspective","from my perspective","in my view","in this view","in my opinion","in our opinion","to my knowledge", "tend to", "tends to"]
assertive = {'claim': 19, 'hypothesize': 28, 'presume': 63, 'figure': 8, 'predict': 36, 'hint': 27, 'prophesy': 37, 'insist': 31, 'testify': 46, 'imply': 29, 'vow': 49, 'expect': 3, 'deduce': 60, 'seem': 6, 'allege': 12, 'guarantee': 26, 'contend': 20, 'guess': 5, 'point out': 35, 'appear': 7, 'acknowledge': 9, 'suggest': 44, 'explain': 24, 'certify': 17, 'divulge': 22, 'write': 50, 'indicate': 30, 'charge': 18, 'swear': 45, 'suspect': 65, 'emphasize': 23, 'certain': 53, 'answer': 13, 'reply': 40, 'postulate': 38, 'surmise': 64, 'hope': 62, 'sure': 54, 'intimate': 32, 'agree': 51, 'assert': 15, 'mention': 34, 'state': 43, 'decide': 59, 'imagine': 4, 'report': 41, 'estimate': 61, 'believe': 1, 'calculate': 58, 'remark': 39, 'theorize': 47, 'evident': 57, 'affirm': 11, 'obvious': 56, 'clear': 55, 'grant': 25, 'say': 42, 'think': 0, 'afraid': 52, 'assure': 16, 'admit': 10, 'maintain': 33, 'suppose': 2, 'verify': 48, 'argue': 14, 'declare': 21}
factive = {'relevant': 22, 'regret': 2, 'discover': 4, 'see': 13, 'odd': 19, 'forget': 3, 'interesting': 21, 'suffice': 16, 'note': 6, 'strange': 20, 'sorry': 23, 'notice': 7, 'perceive': 9, 'resent': 14, 'observe': 8, 'know': 0, 'exciting': 24, 'realize': 1, 'care': 18, 'reveal': 12, 'remember': 11, 'recall': 10, 'bother': 17, 'learn': 5, 'amuse': 15}
factive_phrase = ["find out", "make sense", "found out", "makes sense", "made sense", "finds out"]
propose_keywords = {"$": 0, ".":1,"?":2,"could":3,"middle":4,"meet":5,"go":6,"deal":7,"come":8,"would":9,"ask":10,"will":11,"throw":12,"pick":13}
dominance, valence, arousal = read_dominance_arousal_valence.get_dominance_valence_arousal()
greetings = ["greetings", "hi", "hello", "yo", "hey", "howdy", "sup", "hiya", "how's it going", "how are you", "what's up", "how's everything", "how's your day", "nice to meet you", "good morning", "good afternoon", "good evening"]
apology = ["apologize", "apology", "my bad", "my fault", "my mistake", "my apologies"]
gratitute = ["thank", "grateful", "thankful", "thanks", "appreciate"]

first_person_singular = ["i", "me", "mine", "my"]
first_person_plural = ["we", "our", "us", "ours"]
third_person_singular = ["he","she","it","his","her","him"]
third_person_plural = ["them","they","their"]

def extract_acts(dialog):	
	strategies = list()

	
	lexicon_list = list()
	total_dialogss = 0

	positive_text = ""
	negative_text = ""
	strategy_embedding_text = ""

	dialog_index = 0

	lemmatizer = WordNetLemmatizer()
	positive = 0
	negative = 0
	total_uterance = 0
	pre_complex_features_index = 0
	example_arousal = list()
	example_arousal_score = list()

	propose_hedge = 0
	propose_count = 0
	hedge_count = 0

	liwc_authenticity_text = list()
	#automatically label complex labels
	# complex_features = list()
	
	#rule-based recommendation system
	majority_rules = dict()

	#complex feature calculator
	
	pre_complex_features = list()
	# des_classfier = joblib.load('/projects/tir1/users/rjoshi2/negotiation/yiheng_negotiation/evaluation/Classifier_With_Auto_Labeling/models/des.pkl')
	# infer_classfier = joblib.load('/projects/tir1/users/rjoshi2/negotiation/yiheng_negotiation/evaluation/Classifier_With_Auto_Labeling/models/des.pkl')
	# pata_classfier = joblib.load('/projects/tir1/users/rjoshi2/negotiation/yiheng_negotiation/evaluation/Classifier_With_Auto_Labeling/models/des.pkl')
	# propose_classfier = joblib.load('/projects/tir1/users/rjoshi2/negotiation/yiheng_negotiation/evaluation/Classifier_With_Auto_Labeling/models/des.pkl')
	# des_classfier = joblib.load(curr_file_path + 'des.pkl')
	# infer_classfier = joblib.load(curr_file_path + 'infer.pkl')
	# pata_classfier = joblib.load(curr_file_path + 'pata.pkl')
	# propose_classfier = joblib.load(curr_file_path + 'propose.pkl')
	categories = Counter()
	uter_index_overall = 0
	variance_examples_labels = {"seller_neg_sentiment":list(),"seller_pos_sentiment":list(),"first_person_plural_count_seller":list(),"first_person_singular_count_seller":list(),"third_person_singular_seller":list(),"third_person_plural_seller":list(),"seller_propose":list(),"hedge_count_seller":list(),"factive_count_seller":list(),"who_propose":list(),"seller_trade_in":list(),"sg_concern":list(),"liwc_certainty":list(),"liwc_informal":list(),"politeness_seller_please":list(),"politeness_seller_gratitude":list(),"politeness_seller_please_s":list(),"ap_des":list(),"ap_pata":list(),"ap_infer":list(),"family":list(),"friend":list(),"politeness_seller_greet":list()}
	variance_examples = list()

	#recommendation system, each set of feature represents each uterance
	recommendation_data = list()
	recommendation_feature_mapping = {"seller_neg_sentiment":0,"seller_pos_sentiment":1,"buyer_neg_sentiment":2,"buyer_pos_sentiment":3,"first_person_plural_count_seller":4,"first_person_singular_count_seller":5,"first_person_plural_count_buyer":6,"first_person_singular_count_buyer":7,"third_person_singular_seller":8,"third_person_plural_seller":9,"third_person_singular_buyer":10,"third_person_plural_buyer":11,"number_of_diff_dic_pos":12,"number_of_diff_dic_neg":13,"buyer_propose":14,"seller_propose":15,"hedge_count_seller":16,"hedge_count_buyer":17,"assertive_count_seller":18,"assertive_count_buyer":19,"factive_count_seller":20,"factive_count_buyer":21,"who_propose":22,"seller_trade_in":23,"personal_concern_seller":24,"sg_concern":25,"liwc_certainty":26,"liwc_informal":27,"politeness_seller_please":28,"politeness_seller_gratitude":29,"politeness_seller_please_s":30,"ap_des":31,"ap_pata":32,"ap_infer":33,"family":34,"friend":35,"politeness_buyer_please":36,"politeness_buyer_gratitude":37,"politeness_buyer_please_s":38,"politeness_seller_greet":39,"politeness_buyer_greet":40}
	dialog_length = list()
	recommendation_raw_utterance = list()
	recommendation_product_description = list()
	sequence_of_strategy = list()
	
	#ngram = ""
	ngram_dic = json.load(open(curr_file_path + "ngram_dic_cata"))

	fine_intents = list()

	total_dialogss += 1
	# if "<selle>" not in dialog:
	# 	continue
	# if "<noise>" in dialog:
	# 	continue
	# if "<accept>" not in dialog:
	# 	continue

	#recommendation system
	recommendation_raw_utterance_tmp = list()
	strategy_sequences = list()

	price = dialog["scenario"]["kbs"][1]["personal"]["Target"]
	target = dialog["scenario"]["kbs"][0]["personal"]["Target"]

	complex_features_tmp = list()

	tmp = list()
	tmp_complex = [0,0,0,0,0]
	
	#ngram_features = [0]*len(ngram_dic)
	first_person_plural_count_buyer = 0
	first_person_singular_count_buyer = 0
	first_person_plural_count_seller = 0
	first_person_singular_count_seller = 0
	third_person_plural_buyer = 0
	third_person_singular_buyer = 0
	third_person_singular_seller = 0
	third_person_plural_seller = 0

	dominance_avg_seller = 0.0
	dominance_count_seller = 0
	dominance_avg_buyer = 0.0
	dominance_count_buyer = 0
	valence_avg_buyer = 0.0
	arousal_avg_buyer = 0.0
	valence_avg_seller = 0.0
	arousal_avg_seller = 0.0
	example_arousal_tmp = list()

	number_of_diff_dic_pos = 0
	number_of_diff_dic_neg = 0
	
	total_words_seller = 0
	total_words_buyer = 0
	total_uterance_seller = 0
	total_uterance_buyer = 0
	final = 0
	
	buyer_pos_sentiment = 0
	buyer_neg_sentiment = 0
	seller_pos_sentiment = 0
	seller_neg_sentiment = 0

	greetings_seller = 0
	sg_concern = 0
	politeness_seller_gratitude = 0.0
	politeness_seller_please = 0.0
	politeness_seller_apology = 0.0
	politeness_seller_greetings = 0.0
	politeness_seller_please_s = 0.0
	politeness_buyer_gratitude = 0.0
	politeness_buyer_please = 0.0
	politeness_buyer_apology = 0.0
	politeness_buyer_greetings = 0.0
	politeness_buyer_please_s = 0.0

	politeness_buyer = 0.0
	social_distance_seller = 0.0
	social_distance_count = 0.0
	social_distance_buyer = 0.0
	social_distance_count_buyer = 0.0
	personal_concern_seller = 0
	personal_concern_buyer = 0
	greetings_buyer = 0
	factive_count_seller = 0.0
	factive_count_buyer = 0.0
	hedge_count_seller = 0.0
	hedge_count_buyer = 0.0
	assertive_count_seller = 0.0
	assertive_count_buyer = 0.0
	buyer_first_price = 0.0
	seller_first_price = 0.0
	first_price = True
	_first_price = True
	buyer_propose = 0
	seller_propose = 0
	who_propose = 0
	who_propose_visit = True
	seller_trade_in = 0
	seller_deliver = 0
	buyer_trade_in = 0
	buyer_ask_trade_in = 0
	buyer_reject = 0
	stat_tmp = list()
	liwc_authenticity = 0.0
	liwc_informal = 0.0
	liwc_certainty = 0

	
	propose_hedge_tmp = 0
	propose_count_tmp = 0
	past_tense = 0
	
	uters = list()
	for event in dialog["events"]:
		if event["agent"] == 1:
			agent = "<selle>"
		else:
			agent = "<buyer>"
		if event["action"] == "message":
			uters.append(agent + " " + event["data"])
		elif event["action"] == "accept":
			uters.append(agent + " " +"<accept>")
		elif event["action"] == "reject":
			uters.append(agent + " " +"<reject>")
		elif event["action"] == "offer":
			uters.append(agent + " " +"<offer " + str(event["data"]["price"]) + " >")
		elif event["action"] == "quit":
			uters.append(agent + " " +"<quit>")


	#get rid of noise posts
	number_of_uter = len(uters)
	# if number_of_uter <= 3:
	# 	continue


	uter_index = 0
	portion_index = 1
	buyer_propose_visit = True
	seller_propose_visit = True
	previous = ""
	vocab_tmp = list()
	tmp_strategies = list()
	recommendation_data_uter_cumu = [0.0]*len(recommendation_feature_mapping)
	strategy_embedding_text_dialog = ""
	tmp_strategies_embedding_text = ""


	fine_intents = list()
	fine_intents.append(["<start>"])
	bag_of_strategies = []

	for u_index in range(len(uters)):
		fine_intents.append([])
		uter = uters[u_index]
		keywords = dict()
		tmp_strategy_sequences = list()
		
		o_propose_visit = False

		recommendation_data_uter = [0]*len(recommendation_feature_mapping)
		previous_strategies_embedding = tmp_strategies_embedding_text
		tmp_strategies_embedding_text = uter
		tmp_strategies.append([uter])

		if "<buyer>" in uter and "<offer " not in uter and "<accept>" not in uter:
			#tmp_strategy_sequences.append("<buyer>")
			if ("pick it up" in uter or "pick up" in uter):
				buyer_trade_in += 1
				fine_intents[-1].append("<buyer_trade_in>")
			if ("throw in" in uter or "throwing in" in uter) and ("?" in uter or "if" in uter):
				buyer_ask_trade_in = 1
				fine_intents[-1].append("<buyer_trade_in>")

			buyer_propose_visit = True
			if len(re.findall(r"\d+", uter)) > 0 or len(re.findall(r"[0-9]+,[0-9]+", uter)) > 0:
				for possible_price in re.findall(r"\d+", uter) + re.findall(r"[0-9]+,[0-9]+", uter):
					possible_price = possible_price.replace(",", "")
					if 1.2 	 > float(possible_price)/float(target) > 0.7 and float(possible_price) != float(price) and abs(float(possible_price) - float(target)) < abs(float(buyer_first_price) - float(target)):
						if who_propose_visit:
							who_propose = 0
							tmp_strategies[-1].append("<Wait_For_Buyer_Propose>")
							who_propose_visit = False
						if buyer_propose_visit:
							buyer_propose += 1
							tmp_strategy_sequences.append("<buyer_propose>")
							fine_intents[-1].append("<buyer_propose>")
							buyer_propose_visit = False
							recommendation_data_uter[recommendation_feature_mapping["buyer_propose"]] = 1
							tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(possible_price, " <buyer_propose> ")

						if first_price:
							buyer_first_price = float(possible_price)
				first_price = False
				if buyer_first_price == 0.0:
					first_price = True

			if buyer_propose_visit and (("lowest" in uter and (("?" in uter) or ("what" in uter))) or ("price" in uter and "high" in uter) or ("price" in uter and "lower" in uter)):
				buyer_reject += 1

			previous_word = ""
			word_tokenized = word_tokenize(uter)

			#uterrance wise analysis
			for greet_i in range(len(greetings)):
				if greet_i <= 7:
					if greetings[greet_i] in word_tokenized:
						politeness_buyer_greetings += 1
						recommendation_data_uter[recommendation_feature_mapping["politeness_buyer_greet"]] = 1
						tmp_strategy_sequences.append("<politeness_buyer_greet>")
						fine_intents[-1].append("<politeness_buyer_greet>")
						tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + greetings[greet_i], " <politeness_buyer_greet> ")
				else:
					if greetings[greet_i] in uter:
						politeness_buyer_greetings += 1
						recommendation_data_uter[recommendation_feature_mapping["politeness_buyer_greet"]] = 1
						tmp_strategy_sequences.append("<politeness_buyer_greet>")
						fine_intents[-1].append("<politeness_buyer_greet>")
						tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + greetings[greet_i], " <politeness_buyer_greet> ")

			for grad_i in range(len(gratitute)):
				if gratitute[grad_i] in word_tokenized:
					politeness_buyer_gratitude += 1
					recommendation_data_uter[recommendation_feature_mapping["politeness_buyer_gratitude"]] = 1
					tmp_strategy_sequences.append("<politeness_buyer_gratitude>")
					fine_intents[-1].append("<politeness_buyer_gratitude>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + gratitute[grad_i], " <politeness_buyer_gratitude> ")

			please_index = -1
			for word_i in range(len(word_tokenized)):
				if word_tokenized[word_i] == "please" or word_tokenized[word_i] == "pls":
					please_index = word_i
					break
			if please_index != -1:
				if word_tokenized[please_index-1] != ">" and word_tokenized[please_index-1] != "." and word_tokenized[please_index-1] != "?" and word_tokenized[please_index-1] != "!":
					politeness_buyer_please += 1
					recommendation_data_uter[recommendation_feature_mapping["politeness_buyer_please"]] = 1
					tmp_strategy_sequences.append("<politeness_buyer_please>")
					fine_intents[-1].append("<politeness_buyer_please>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" please", " <politeness_buyer_please> ").replace("pls", "<politeness_buyer_please>")
				else:
					politeness_buyer_please_s += 1
					recommendation_data_uter[recommendation_feature_mapping["politeness_buyer_please_s"]] = 1
					tmp_strategy_sequences.append("<politeness_buyer_please_s>")
					fine_intents[-1].append("<politeness_buyer_please>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" please", " <politeness_buyer_please_s> ").replace("pls", "<politeness_buyer_please_s>")


			for word in word_tokenized:
				word = lemmatizer.lemmatize(word)
				if word in liwc_friend and (previous_word != "your" and previous_word != "ur"):
					social_distance_buyer += 1.0
					social_distance_count_buyer += 1.0
				if word in liwc_family and (previous_word != "your" and previous_word != "ur"):
					social_distance_buyer += 0.0
					social_distance_count_buyer += 1.0
				if word in factive:
					factive_count_buyer += 1
					recommendation_data_uter[recommendation_feature_mapping["factive_count_buyer"]] = 1
					tmp_strategy_sequences.append("<factive_count_buyer>")
					fine_intents[-1].append("<factive_count_buyer>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <factive_count_buyer> ")

				if word in first_person_singular:
					first_person_singular_count_buyer += 1
					recommendation_data_uter[recommendation_feature_mapping["first_person_singular_count_buyer"]] = 1
					tmp_strategy_sequences.append("<first_person_singular_count_buyer>")
					fine_intents[-1].append("<first_person_singular_count_buyer>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word +" ", " <first_person_singular_count_buyer> ")
				elif word in first_person_plural:
					first_person_plural_count_buyer += 1
					recommendation_data_uter[recommendation_feature_mapping["first_person_plural_count_buyer"]] = 1
					tmp_strategy_sequences.append("<first_person_plural_count_buyer>")
					fine_intents[-1].append("<first_person_plural_count_buyer>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <first_person_plural_count_buyer> ")
				elif word in third_person_plural:
					third_person_plural_buyer += 1
					recommendation_data_uter[recommendation_feature_mapping["third_person_plural_buyer"]] = 1
					tmp_strategy_sequences.append("<third_person_plural_buyer>")
					fine_intents[-1].append("<third_person_plural_buyer>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <third_person_plural_buyer> ")
				elif word in third_person_singular:
					third_person_singular_buyer += 1
					recommendation_data_uter[recommendation_feature_mapping["third_person_singular_buyer"]] = 1
					tmp_strategy_sequences.append("<third_person_singular_buyer>")
					fine_intents[-1].append("<third_person_singular_buyer>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <third_person_singular_buyer> ")
				if word in assertive:
					assertive_count_buyer += 1
					recommendation_data_uter[recommendation_feature_mapping["assertive_count_buyer"]] = 1
					tmp_strategy_sequences.append("<assertive_count_buyer>")
					fine_intents[-1].append("<assertive_count_buyer>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <assertive_count_buyer> ")
				if word in hedges_word:
					if _first_price and first_price:
						example_arousal_tmp.append(word + "," + "N/A" + "," + uter.replace(",", ";"))
					hedge_count_buyer += 1
					recommendation_data_uter[recommendation_feature_mapping["hedge_count_buyer"]] = 1
					tmp_strategy_sequences.append("<hedge_count_buyer>")
					fine_intents[-1].append("<hedge_count_buyer>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <hedge_count_buyer> ")
				if word in pos:
					buyer_pos_sentiment += 1
					recommendation_data_uter[recommendation_feature_mapping["buyer_pos_sentiment"]] = 1
					tmp_strategy_sequences.append("<buyer_pos_sentiment>")
					fine_intents[-1].append("<buyer_pos_sentiment>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <buyer_pos_sentiment> ")
				if word in neg:
					buyer_neg_sentiment += 1
					recommendation_data_uter[recommendation_feature_mapping["buyer_neg_sentiment"]] = 1
					tmp_strategy_sequences.append("<buyer_neg_sentiment>")
					fine_intents[-1].append("<buyer_neg_sentiment>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <buyer_neg_sentiment> ")
				if word in dominance:
					dominance_count_buyer += 1
					dominance_avg_buyer += dominance[word]
					valence_avg_buyer += valence[word]
					arousal_avg_buyer += arousal[word]
				total_words_buyer += 1
				stat_tmp.append(word)
				vocab_tmp.append(word)
				previous_word = word
			total_uterance_buyer += 1
			recommendation_data_uter_cumu = [a + b for a, b in zip(recommendation_data_uter_cumu, recommendation_data_uter[:-2])]
			recommendation_raw_utterance_tmp.append(tmp_strategies_embedding_text)
			strategy_sequences.append(tmp_strategy_sequences)

		if "<selle>" in uter and "<offer " not in uter and "<accept>" not in uter:
			variance_examples.append(uter)
			for key in variance_examples_labels:
				variance_examples_labels[key].append(0)

			#tmp_strategy_sequences.append("<selle>")
			if "throw in" in uter or "throwing in" in uter:
				seller_trade_in = 1
				recommendation_data_uter[recommendation_feature_mapping["seller_trade_in"]] = 1
				tmp_strategy_sequences.append("<seller_trade_in>")
				fine_intents[-1].append("<seller_trade_in>")
				tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace("throw in", "<seller_trade_in>").replace("throwing in", "<seller_trade_in>")
				if first_price and _first_price:
					tmp_strategies[-1].append("<1_Trade_In>")
				else:
					tmp_strategies[-1].append("<2_Trade_In>")
				variance_examples_labels["seller_trade_in"][-1] = 1
			if "deliver" in uter:
				recommendation_data_uter[recommendation_feature_mapping["seller_trade_in"]] = 1
				fine_intents[-1].append("<seller_trade_in>")
				seller_deliver += 1

			if len(re.findall(r"\d+", uter)) > 0:
				#seller_propose_visit = True
				for possible_price in re.findall(r"\d+", uter):
					if 1 > float(possible_price)/float(price) > 0.7  and abs(float(possible_price) - float(price)) < abs(float(buyer_first_price) - float(price)):
						if seller_propose_visit:
							seller_propose += 1
							tmp_strategy_sequences.append("<seller_propose>")
							fine_intents[-1].append("<seller_propose>")
							recommendation_data_uter[recommendation_feature_mapping["seller_propose"]] = 1
							tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(possible_price, "<seller_propose>")
							tmp_strategies[-1].append("<Propose_New_Price>")
							seller_propose_visit = False
							variance_examples_labels["seller_propose"][-1] = 1
						if _first_price:
							seller_first_price = float(possible_price)
					if 1 > float(possible_price)/float(price) > 0.5:
						if who_propose_visit:
							who_propose = 1
							recommendation_data_uter[recommendation_feature_mapping["who_propose"]] = 1
							who_propose_visit = False
							variance_examples_labels["who_propose"][-1] = 1
				_first_price = False
				if seller_first_price == 0.0:
					_first_price = True

			word_tokenized = word_tokenize(uter)

			# TODO (INSERT CODE HERE FOR CLASSIFIER BASED STRATEGIES)

			#uterrance wise analysis
			for greet_i in range(len(greetings)):
				if greet_i <= 7:
					if greetings[greet_i] in word_tokenized:
						politeness_seller_greetings += 1
						recommendation_data_uter[recommendation_feature_mapping["politeness_seller_greet"]] = 1
						tmp_strategy_sequences.append("<politeness_seller_greet>")
						fine_intents[-1].append("<politeness_seller_greet>")
						tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + greetings[greet_i], " <politeness_seller_greet> ")
						if first_price and _first_price:
							tmp_strategies[-1].append("<1_Greetings>")
						else:
							tmp_strategies[-1].append("<2_Greetings>")
						variance_examples_labels["politeness_seller_greet"][-1] = 1
				else:
					if greetings[greet_i] in uter:
						politeness_seller_greetings += 1
						recommendation_data_uter[recommendation_feature_mapping["politeness_seller_greet"]] = 1
						tmp_strategy_sequences.append("<politeness_seller_greet>")
						fine_intents[-1].append("<politeness_seller_greet>")
						tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + greetings[greet_i], " <politeness_seller_greet> ")
						if first_price and _first_price:
							tmp_strategies[-1].append("<1_Greetings>")
						else:
							tmp_strategies[-1].append("<2_Greetings>")
						variance_examples_labels["politeness_seller_greet"][-1] = 1
			
			for sorry_i in range(len(apology)):
				if sorry_i <= 1:
					if apology[sorry_i] in word_tokenized:
						politeness_seller_apology += 1
						if first_price and _first_price:
							tmp_strategies[-1].append("<1_Apology>")
						else:
							tmp_strategies[-1].append("<2_Apology>")
				else:
					if apology[sorry_i] in uter:
						politeness_seller_apology += 1
						if first_price and _first_price:
							tmp_strategies[-1].append("<1_Apology>")
						else:
							tmp_strategies[-1].append("<2_Apology>")
					
				
			for grad_i in range(len(gratitute)):
				if gratitute[grad_i] in word_tokenized:
					politeness_seller_gratitude += 1
					recommendation_data_uter[recommendation_feature_mapping["politeness_seller_gratitude"]] = 1
					tmp_strategy_sequences.append("<politeness_seller_gratitude>")
					fine_intents[-1].append("<politeness_seller_gratitude>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + gratitute[grad_i], " <politeness_seller_gratitude> ")
					if first_price and _first_price:
						tmp_strategies[-1].append("<1_gratitude>")
					else:
						tmp_strategies[-1].append("<2_gratitude>")
					variance_examples_labels["politeness_seller_gratitude"][-1] = 1

			please_index = -1
			for word_i in range(len(word_tokenized)):
				if word_tokenized[word_i] == "please" or word_tokenized[word_i] == "pls":
					please_index = word_i
					break
			if please_index != -1:
				if word_tokenized[please_index-1] != ">" and word_tokenized[please_index-1] != "." and word_tokenized[please_index-1] != "?" and word_tokenized[please_index-1] != "!":
					politeness_seller_please += 1
					recommendation_data_uter[recommendation_feature_mapping["politeness_seller_please"]] = 1
					tmp_strategy_sequences.append("<politeness_seller_please>")
					fine_intents[-1].append("<politeness_seller_please>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace("please", "<politeness_seller_please>").replace("pls", "<politeness_seller_please>")
					if first_price and _first_price:
						tmp_strategies[-1].append("<1_Please>")
					else:
						tmp_strategies[-1].append("<2_Please>")
					variance_examples_labels["politeness_seller_please"][-1] = 1
				else:
					politeness_seller_please_s += 1
					recommendation_data_uter[recommendation_feature_mapping["politeness_seller_please_s"]] = 1
					tmp_strategy_sequences.append("<politeness_seller_please_s>")
					fine_intents[-1].append("<politeness_seller_please_s>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace("please", "<politeness_seller_please_s>").replace("pls", "<politeness_seller_please_s>")
					if first_price and _first_price:
						tmp_strategies[-1].append("<1_Please_Start>")
					else:
						tmp_strategies[-1].append("<2_Please_Start>")
					variance_examples_labels["politeness_seller_please_s"][-1] = 1

			previous_word = ""
			for word in word_tokenized:
				word = lemmatizer.lemmatize(word)
				if word in liwc_informal_dic and word != "ha" and word != "yes" and word != "like" and word != "absolutely" and word != "agree" and word != "ok":
					if word != "well":
						liwc_informal += 1
						recommendation_data_uter[recommendation_feature_mapping["liwc_informal"]] = 1
						tmp_strategy_sequences.append("<liwc_informal>")
						fine_intents[-1].append("<liwc_informal>")
						tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <liwc_informal> ")
						if first_price and _first_price:
							tmp_strategies[-1].append("<1_Informal_Word>")
						else:
							tmp_strategies[-1].append("<2_Informal_Word>")
						variance_examples_labels["liwc_informal"][-1] = 1
					else:
						if previous_word == ">":
							if first_price and _first_price:
								tmp_strategies[-1].append("<1_Informal_Word>")
							else:
								tmp_strategies[-1].append("<2_Informal_Word>")
							liwc_informal += 1
							recommendation_data_uter[recommendation_feature_mapping["liwc_informal"]] = 1
							tmp_strategy_sequences.append("<liwc_informal>")
							fine_intents[-1].append("<liwc_informal>")
							tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <liwc_informal> ")
							variance_examples_labels["liwc_informal"][-1] = 1
				if word in liwc_certain and word != "certain":
					if word == "sure":
						if previous_word != "not" and previous_word != ">" and "?" not in uter:
							liwc_certainty += 1
							recommendation_data_uter[recommendation_feature_mapping["liwc_certainty"]] = 1
							tmp_strategy_sequences.append("<liwc_certainty>")
							fine_intents[-1].append("<liwc_certainty>")
							tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <liwc_certainty> ")
							if first_price and _first_price:
								tmp_strategies[-1].append("<1_Certain_Word>")
							else:
								tmp_strategies[-1].append("<2_Certain_Word>")
							variance_examples_labels["liwc_certainty"][-1] = 1
							# if not (first_price and _first_price):
							# 	print word + "," + uter.replace(",","")
					else:
						liwc_certainty += 1
						tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <liwc_certainty> ")
						recommendation_data_uter[recommendation_feature_mapping["liwc_certainty"]] = 1
						tmp_strategy_sequences.append("<liwc_certainty>")
						fine_intents[-1].append("<liwc_certainty>")
						if first_price and _first_price:
							tmp_strategies[-1].append("<1_Certain_Word>")
						else:
							tmp_strategies[-1].append("<2_Certain_Word>")
						variance_examples_labels["liwc_certainty"][-1] = 1
						# if not (first_price and _first_price):
						# 	print word + "," + uter.replace(",","")
				if word in liwc_friend and (previous_word != "your" and previous_word != "ur") and not word.startswith("bud"):
					social_distance_seller += 1.0
					social_distance_count += 1.0
					recommendation_data_uter[recommendation_feature_mapping["friend"]] = 1
					tmp_strategy_sequences.append("<friend>")
					fine_intents[-1].append("<friend>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <friend> ")
					if first_price and _first_price:
						tmp_strategies[-1].append("<1_Friend_Word>")
					else:
						tmp_strategies[-1].append("<2_Friend_Word>")
					variance_examples_labels["friend"][-1] = 1
				if word in liwc_family and (previous_word != "your" and previous_word != "ur"):
					if word == "family":
						if previous_word == "my":
							social_distance_seller += 0.0
							recommendation_data_uter[recommendation_feature_mapping["family"]] = 1
							tmp_strategy_sequences.append("<family>")
							fine_intents[-1].append("<family>")
							tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <family> ")
							social_distance_count += 1.0
							if first_price and _first_price:
								tmp_strategies[-1].append("<1_Family>")
							else:
								tmp_strategies[-1].append("<2_Family>")
							variance_examples_labels["family"][-1] = 1
					else:
						social_distance_seller += 0.0
						recommendation_data_uter[recommendation_feature_mapping["family"]] = 1
						tmp_strategy_sequences.append("<family>")
						fine_intents[-1].append("<family>")
						tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <family> ")
						social_distance_count += 1.0
						if first_price and _first_price:
							tmp_strategies[-1].append("<1_Family>")
						else:
							tmp_strategies[-1].append("<2_Family>")
						variance_examples_labels["family"][-1] = 1
				if word in liwc_personal_concern:
					personal_concern_seller += 1
					recommendation_data_uter[recommendation_feature_mapping["personal_concern_seller"]] = 1
					tmp_strategy_sequences.append("<personal_concern_seller>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <personal_concern_seller> ")
				if word in factive:
					keywords["factive_count_seller"] = word
					factive_count_seller += 1
					recommendation_data_uter[recommendation_feature_mapping["factive_count_seller"]] = 1
					tmp_strategy_sequences.append("<factive_count_seller>")
					fine_intents[-1].append("<factive_count_seller>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <factive_count_seller> ")
					variance_examples_labels["factive_count_seller"][-1] = 1
				if word in first_person_singular:
					first_person_singular_count_seller += 1
					recommendation_data_uter[recommendation_feature_mapping["first_person_singular_count_seller"]] = 1
					tmp_strategy_sequences.append("<first_person_singular_count_seller>")
					fine_intents[-1].append("<first_person_singular_count_seller>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word +" ", " <first_person_singular_count_seller> ")
					variance_examples_labels["first_person_singular_count_seller"][-1] = 1
				elif word in first_person_plural:
					first_person_plural_count_seller += 1
					recommendation_data_uter[recommendation_feature_mapping["first_person_plural_count_seller"]] = 1
					tmp_strategy_sequences.append("<first_person_plural_count_seller>")
					fine_intents[-1].append("<first_person_plural_count_seller>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <first_person_plural_count_seller> ")
					variance_examples_labels["first_person_plural_count_seller"][-1] = 1
				elif word in third_person_plural:
					third_person_plural_seller += 1
					recommendation_data_uter[recommendation_feature_mapping["third_person_plural_seller"]] = 1
					tmp_strategy_sequences.append("<third_person_plural_seller>")
					fine_intents[-1].append("<third_person_plural_seller>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <third_person_plural_seller> ")
					variance_examples_labels["third_person_plural_seller"][-1] = 1
				elif word in third_person_singular:
					third_person_singular_seller += 1
					recommendation_data_uter[recommendation_feature_mapping["third_person_singular_seller"]] = 1
					tmp_strategy_sequences.append("<third_person_singular_seller>")
					fine_intents[-1].append("<third_person_singular_seller>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <third_person_singular_seller> ")
					variance_examples_labels["third_person_singular_seller"][-1] = 1
				if word in dominance:
					dominance_count_seller += 1
					dominance_avg_seller += dominance[word]
					valence_avg_seller += valence[word]
					arousal_avg_seller += arousal[word]
				if word in assertive:
					keywords["assertive_count_seller"] = word
					assertive_count_seller += 1
					recommendation_data_uter[recommendation_feature_mapping["assertive_count_seller"]] = 1
					tmp_strategy_sequences.append("<assertive_count_seller>")
					fine_intents[-1].append("<assertive_count_seller>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <assertive_count_seller> ")
				if word in hedges_word:
					#print word + "," + uter.replace(",","")
					keywords["hedge_count_seller"] = word
					recommendation_data_uter[recommendation_feature_mapping["hedge_count_seller"]] = 1
					tmp_strategy_sequences.append("<hedge_count_seller>")
					fine_intents[-1].append("<hedge_count_seller>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <hedge_count_seller> ")
					hedge_count_seller += 1
					if o_propose_visit:
						propose_hedge_tmp += 1
					variance_examples_labels["hedge_count_seller"][-1] = 1
				if word in pos:
					seller_pos_sentiment += 1
					recommendation_data_uter[recommendation_feature_mapping["seller_pos_sentiment"]] = 1
					tmp_strategy_sequences.append("<seller_pos_sentiment>")
					fine_intents[-1].append("<seller_pos_sentiment>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <seller_pos_sentiment> ")
					variance_examples_labels["seller_pos_sentiment"][-1] = 1
				if word in neg:
					seller_neg_sentiment += 1
					recommendation_data_uter[recommendation_feature_mapping["seller_neg_sentiment"]] = 1
					tmp_strategy_sequences.append("<seller_neg_sentiment>")
					fine_intents[-1].append("<seller_neg_sentiment>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <seller_neg_sentiment> ")
					variance_examples_labels["seller_neg_sentiment"][-1] = 1
				if word in lexicon_diff_dic_pos:
					number_of_diff_dic_pos += 1
					recommendation_data_uter[recommendation_feature_mapping["number_of_diff_dic_pos"]] = 1
					tmp_strategy_sequences.append("<number_of_diff_dic_pos>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <number_of_diff_dic_pos> ")
				if word in lexicon_diff_dic_neg:
					number_of_diff_dic_neg += 1
					recommendation_data_uter[recommendation_feature_mapping["number_of_diff_dic_neg"]] = 1
					tmp_strategy_sequences.append("<number_of_diff_dic_neg>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <number_of_diff_dic_neg> ")
				total_words_seller += 1
				stat_tmp.append(word)
				vocab_tmp.append(word)
				previous_word = word 
			total_uterance_seller += 1
			recommendation_data_uter_cumu = [a + b for a, b in zip(recommendation_data_uter_cumu, recommendation_data_uter[:-2])]
			recommendation_raw_utterance_tmp.append(tmp_strategies_embedding_text)
			strategy_sequences.append(tmp_strategy_sequences)

			if len(strategy_sequences) > 1:
				if ",".join(strategy_sequences[-2]) not in majority_rules:
					majority_rules[",".join(strategy_sequences[-2])] = Counter()
					majority_rules[",".join(strategy_sequences[-2])][",".join(strategy_sequences[-1])]+=1
				else:
					majority_rules[",".join(strategy_sequences[-2])][",".join(strategy_sequences[-1])]+=1

			recommendation_template_tmp = [1,0] + recommendation_data_uter[:-2] 
			key = "".join([str(int(a)) for a in recommendation_template_tmp])
			if key not in recommendation_template and keywords:
				recommendation_template[key] = [uter, keywords]


		uter_index += 1
		
		bag_of_strategies.append(recommendation_data_uter)							 
		strategy_embedding_text_dialog += previous_strategies_embedding + " " + tmp_strategies_embedding_text + "\n"
		fine_intents[-1].append("<end>")


		# if "<offer " in uter:
		# 	final = re.findall(r'\d+', uter)[0]

		if (not (first_price and _first_price) and portion_index == 1) or (uter_index == number_of_uter): # #uter_index >= portion_index*number_of_uter/4.0 and portion_index <= 4:
			if len(tmp) == 0 and uter_index == number_of_uter:
				tmp = [0.0]*feature_size
				tmp[-10] = who_propose

			# if portion_index == 1:
			# 	stage_1_stat += vocab_tmp
			# 	vocab_tmp = list()
			# else:
			# 	stage_2_stat += vocab_tmp

			portion_index += 1
			tmp += tmp_complex

			#sentiment features
			tmp.append(seller_neg_sentiment)
			tmp.append(seller_pos_sentiment)
			tmp.append(buyer_neg_sentiment)
			tmp.append(buyer_pos_sentiment)

			#stubborn features
			if (dominance_count_buyer != 0):
				tmp.append(dominance_avg_buyer/dominance_count_buyer)
				tmp.append(arousal_avg_buyer/dominance_count_buyer)
			else:
				tmp.append(0)
				tmp.append(0)
			if (dominance_count_seller != 0):
				tmp.append(dominance_avg_seller/dominance_count_seller)
				tmp.append(arousal_avg_seller/dominance_count_seller)
			else:
				tmp.append(0)
				tmp.append(0)

			tmp.append(first_person_plural_count_seller)
			tmp.append(first_person_singular_count_seller)
			tmp.append(first_person_plural_count_buyer)
			tmp.append(first_person_singular_count_buyer)
			tmp.append(third_person_singular_seller)
			tmp.append(third_person_plural_seller)
			tmp.append(third_person_singular_buyer)
			tmp.append(third_person_plural_buyer)
			tmp.append(number_of_diff_dic_pos)
			tmp.append(number_of_diff_dic_neg)

			#most informative ones	
			if total_uterance_seller == 0:
				tmp.append(0)
			else:
				tmp.append(total_words_seller/total_uterance_seller)
			if total_uterance_buyer == 0:
				tmp.append(0)
			else:
				tmp.append(total_words_buyer/total_uterance_buyer)
			tmp.append(buyer_propose)
			tmp.append(seller_propose)


			tmp.append(float(buyer_first_price))
			tmp.append(float(seller_first_price))
			tmp.append(float(price))
			tmp.append((float(buyer_first_price) - float(price))/float(price))

			tmp.append(hedge_count_seller)
			tmp.append(hedge_count_buyer)
			tmp.append(assertive_count_seller)
			tmp.append(assertive_count_buyer)
			tmp.append(factive_count_seller)
			tmp.append(factive_count_buyer)

			
			tmp.append(who_propose)
			tmp.append(seller_trade_in)
			tmp.append(personal_concern_seller)
			tmp.append(sg_concern)
			if social_distance_count != 0:
				tmp.append(social_distance_seller/social_distance_count)
			else:
				tmp.append(2.0)
			tmp.append(liwc_certainty)
			tmp.append(liwc_informal)
			#tmp.append(politeness_seller_apology)
			#tmp.append(politeness_seller_greetings)
			tmp.append(politeness_seller_please)
			tmp.append(politeness_seller_gratitude)
			tmp.append(politeness_seller_please_s)

			if uter_index != number_of_uter:
				tmp_complex = [0,0,0,0,0]
				seller_neg_sentiment = 0.0
				seller_pos_sentiment = 0.0
				buyer_neg_sentiment = 0.0
				buyer_pos_sentiment = 0.0
				dominance_avg_buyer = 0.0
				dominance_count_buyer = 0.0
				arousal_avg_buyer = 0.0
				dominance_count_buyer = 0.0
				dominance_avg_seller = 0.0
				dominance_count_seller = 0.0
				arousal_avg_seller = 0.0
				dominance_count_seller = 0.0
				first_person_plural_count_seller = 0.0
				first_person_singular_count_seller = 0.0
				first_person_plural_count_buyer = 0.0
				first_person_singular_count_buyer = 0.0
				third_person_singular_seller = 0.0
				third_person_plural_seller = 0.0
				third_person_singular_buyer = 0.0
				third_person_plural_buyer = 0.0
				number_of_diff_dic_pos = 0.0
				number_of_diff_dic_neg = 0.0
				total_uterance_seller  = 0.0
				total_words_seller  = 0.0
				total_words_buyer = 0.0
				total_uterance_buyer = 0.0
				buyer_propose  = 0.0
				seller_propose  = 0.0
				hedge_count_seller = 0.0
				hedge_count_buyer = 0.0
				assertive_count_seller = 0.0
				assertive_count_buyer = 0.0
				factive_count_seller = 0.0
				factive_count_buyer = 0.0
				seller_trade_in = 0
				personal_concern_seller = 0
				sg_concern = 0
				social_distance_seller = 0.0
				social_distance_count  = 0.0
				liwc_certainty = 0.0
				liwc_informal = 0.0
				#politeness_seller_apology = 0.0
				#politeness_seller_greetings = 0.0
				politeness_seller_please = 0.0
				politeness_seller_gratitude = 0.0
				politeness_seller_please_s = 0.0

		previous = uter
		uter_index_overall += 1

	return fine_intents, bag_of_strategies


def extract_seq_acts(dialog):	
	strategies = list()
	extracted_seqs = list()

	lexicon_list = list()
	total_dialogss = 0
	
	positive_text = ""
	negative_text = ""
	strategy_embedding_text = ""

	dialog_index = 0

	lemmatizer = WordNetLemmatizer()
	positive = 0
	negative = 0
	total_uterance = 0
	pre_complex_features_index = 0
	example_arousal = list()
	example_arousal_score = list()

	propose_hedge = 0
	propose_count = 0
	hedge_count = 0

	liwc_authenticity_text = list()
	#automatically label complex labels
	# complex_features = list()
	
	#rule-based recommendation system
	majority_rules = dict()

	#complex feature calculator
	pre_complex_features = list()
	

	categories = Counter()
	uter_index_overall = 0
	variance_examples_labels = {"seller_neg_sentiment":list(),"seller_pos_sentiment":list(),"first_person_plural_count_seller":list(),"first_person_singular_count_seller":list(),"third_person_singular_seller":list(),"third_person_plural_seller":list(),"seller_propose":list(),"hedge_count_seller":list(),"factive_count_seller":list(),"who_propose":list(),"seller_trade_in":list(),"sg_concern":list(),"liwc_certainty":list(),"liwc_informal":list(),"politeness_seller_please":list(),"politeness_seller_gratitude":list(),"politeness_seller_please_s":list(),"ap_des":list(),"ap_pata":list(),"ap_infer":list(),"family":list(),"friend":list(),"politeness_seller_greet":list()}
	variance_examples = list()

	#recommendation system, each set of feature represents each uterance
	recommendation_data = list()
	recommendation_feature_mapping = {"seller_neg_sentiment":0,"seller_pos_sentiment":1,"buyer_neg_sentiment":2,"buyer_pos_sentiment":3,"first_person_plural_count_seller":4,"first_person_singular_count_seller":5,"first_person_plural_count_buyer":6,"first_person_singular_count_buyer":7,"third_person_singular_seller":8,"third_person_plural_seller":9,"third_person_singular_buyer":10,"third_person_plural_buyer":11,"number_of_diff_dic_pos":12,"number_of_diff_dic_neg":13,"buyer_propose":14,"seller_propose":15,"hedge_count_seller":16,"hedge_count_buyer":17,"assertive_count_seller":18,"assertive_count_buyer":19,"factive_count_seller":20,"factive_count_buyer":21,"who_propose":22,"seller_trade_in":23,"personal_concern_seller":24,"sg_concern":25,"liwc_certainty":26,"liwc_informal":27,"politeness_seller_please":28,"politeness_seller_gratitude":29,"politeness_seller_please_s":30,"ap_des":31,"ap_pata":32,"ap_infer":33,"family":34,"friend":35,"politeness_buyer_please":36,"politeness_buyer_gratitude":37,"politeness_buyer_please_s":38,"politeness_seller_greet":39,"politeness_buyer_greet":40}
	dialog_length = list()
	recommendation_raw_utterance = list()
	recommendation_product_description = list()
	sequence_of_strategy = list()
	
	#ngram = ""
	ngram_dic = json.load(open(curr_file_path + "ngram_dic_cata"))

	fine_intents = list()

	total_dialogss += 1
	# if "<selle>" not in dialog:
	# 	continue
	# if "<noise>" in dialog:
	# 	continue
	# if "<accept>" not in dialog:
	# 	continue

	#recommendation system
	recommendation_raw_utterance_tmp = list()
	strategy_sequences = list()

	price = dialog["scenario"]["kbs"][1]["personal"]["Target"]
	target = dialog["scenario"]["kbs"][0]["personal"]["Target"]

	complex_features_tmp = list()

	tmp = list()
	tmp_complex = [0,0,0,0,0]
	
	#ngram_features = [0]*len(ngram_dic)
	first_person_plural_count_buyer = 0
	first_person_singular_count_buyer = 0
	first_person_plural_count_seller = 0
	first_person_singular_count_seller = 0
	third_person_plural_buyer = 0
	third_person_singular_buyer = 0
	third_person_singular_seller = 0
	third_person_plural_seller = 0

	dominance_avg_seller = 0.0
	dominance_count_seller = 0
	dominance_avg_buyer = 0.0
	dominance_count_buyer = 0
	valence_avg_buyer = 0.0
	arousal_avg_buyer = 0.0
	valence_avg_seller = 0.0
	arousal_avg_seller = 0.0
	example_arousal_tmp = list()

	number_of_diff_dic_pos = 0
	number_of_diff_dic_neg = 0
	
	total_words_seller = 0
	total_words_buyer = 0
	total_uterance_seller = 0
	total_uterance_buyer = 0
	final = 0
	
	buyer_pos_sentiment = 0
	buyer_neg_sentiment = 0
	seller_pos_sentiment = 0
	seller_neg_sentiment = 0

	greetings_seller = 0
	sg_concern = 0
	politeness_seller_gratitude = 0.0
	politeness_seller_please = 0.0
	politeness_seller_apology = 0.0
	politeness_seller_greetings = 0.0
	politeness_seller_please_s = 0.0
	politeness_buyer_gratitude = 0.0
	politeness_buyer_please = 0.0
	politeness_buyer_apology = 0.0
	politeness_buyer_greetings = 0.0
	politeness_buyer_please_s = 0.0

	politeness_buyer = 0.0
	social_distance_seller = 0.0
	social_distance_count = 0.0
	social_distance_buyer = 0.0
	social_distance_count_buyer = 0.0
	personal_concern_seller = 0
	personal_concern_buyer = 0
	greetings_buyer = 0
	factive_count_seller = 0.0
	factive_count_buyer = 0.0
	hedge_count_seller = 0.0
	hedge_count_buyer = 0.0
	assertive_count_seller = 0.0
	assertive_count_buyer = 0.0
	buyer_first_price = 0.0
	seller_first_price = 0.0
	first_price = True
	_first_price = True
	buyer_propose = 0
	seller_propose = 0
	who_propose = 0
	who_propose_visit = True
	seller_trade_in = 0
	seller_deliver = 0
	buyer_trade_in = 0
	buyer_ask_trade_in = 0
	buyer_reject = 0
	stat_tmp = list()
	liwc_authenticity = 0.0
	liwc_informal = 0.0
	liwc_certainty = 0

	
	propose_hedge_tmp = 0
	propose_count_tmp = 0
	past_tense = 0
	
	uters = list()
	for event in dialog["events"]:
		if event["agent"] == 1:
			agent = "<selle>"
		else:
			agent = "<buyer>"
		if event["action"] == "message":
			uters.append(agent + " " + event["data"])
		elif event["action"] == "accept":
			uters.append(agent + " " +"<accept>")
		elif event["action"] == "reject":
			uters.append(agent + " " +"<reject>")
		elif event["action"] == "offer":
			uters.append(agent + " " +"<offer " + str(event["data"]["price"]) + " >")
		elif event["action"] == "quit":
			uters.append(agent + " " +"<quit>")


	#get rid of noise posts
	number_of_uter = len(uters)
	# if number_of_uter <= 3:
	# 	continue


	uter_index = 0
	portion_index = 1
	buyer_propose_visit = True
	seller_propose_visit = True
	previous = ""
	vocab_tmp = list()
	tmp_strategies = list()
	recommendation_data_uter_cumu = [0.0]*len(recommendation_feature_mapping)
	strategy_embedding_text_dialog = ""
	tmp_strategies_embedding_text = ""


	fine_intents = list()
	fine_intents.append(["<start>"])
	bag_of_strategies = []

	for u_index in range(len(uters)):
		fine_intents.append([])
		uter = uters[u_index]
		uter_split = uter.split(" ")
		extracted_seqs.append([-1] * len(uter_split))
		keywords = dict()
		tmp_strategy_sequences = list()
		word_tokenized = uter_split
		
		o_propose_visit = False

		recommendation_data_uter = [0]*len(recommendation_feature_mapping)
		previous_strategies_embedding = tmp_strategies_embedding_text
		tmp_strategies_embedding_text = uter
		tmp_strategies.append([uter])

		if "<buyer>" in uter and "<offer " not in uter and "<accept>" not in uter:
			#tmp_strategy_sequences.append("<buyer>")
			if ("pick it up" in uter or "pick up" in uter):
				buyer_trade_in += 1
				fine_intents[-1].append("<buyer_trade_in>")
				if "pick" in uter_split:
					extracted_seqs[-1][uter_split.index("pick")] = "<buyer_trade_in>"

			if ("throw in" in uter or "throwing in" in uter) and ("?" in uter or "if" in uter):
				buyer_ask_trade_in = 1
				fine_intents[-1].append("<buyer_trade_in>")
				if "throw" in uter_split:
					extracted_seqs[-1][uter_split.index("throw")] = "<buyer_trade_in>"
				elif "throwing" in uter_split:
					extracted_seqs[-1][uter_split.index("throwing")] = "<buyer_trade_in>"

			buyer_propose_visit = True
			if len(re.findall(r"\d+", uter)) > 0 or len(re.findall(r"[0-9]+,[0-9]+", uter)) > 0:
				for possible_price in re.findall(r"\d+", uter) + re.findall(r"[0-9]+,[0-9]+", uter):
					possible_price = possible_price.replace(",", "")
					if 1.2 	 > float(possible_price)/float(target) > 0.7 and float(possible_price) != float(price) and abs(float(possible_price) - float(target)) < abs(float(buyer_first_price) - float(target)):
						if who_propose_visit:
							who_propose = 0
							tmp_strategies[-1].append("<Wait_For_Buyer_Propose>")
							who_propose_visit = False
						if buyer_propose_visit:
							buyer_propose += 1
							tmp_strategy_sequences.append("<buyer_propose>")
							fine_intents[-1].append("<buyer_propose>")
							if str(possible_price) in uter_split:
								extracted_seqs[-1][uter_split.index(str(possible_price))] = "<buyer_propose>"
							buyer_propose_visit = False
							recommendation_data_uter[recommendation_feature_mapping["buyer_propose"]] = 1
							tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(possible_price, " <buyer_propose> ")

						if first_price:
							buyer_first_price = float(possible_price)
				first_price = False
				if buyer_first_price == 0.0:
					first_price = True

			if buyer_propose_visit and (("lowest" in uter and (("?" in uter) or ("what" in uter))) or ("price" in uter and "high" in uter) or ("price" in uter and "lower" in uter)):
				buyer_reject += 1

			previous_word = ""

			#uterrance wise analysis
			for greet_i in range(len(greetings)):
				if greet_i <= 7:
					if greetings[greet_i] in word_tokenized:
						politeness_buyer_greetings += 1
						recommendation_data_uter[recommendation_feature_mapping["politeness_buyer_greet"]] = 1
						tmp_strategy_sequences.append("<politeness_buyer_greet>")
						fine_intents[-1].append("<politeness_buyer_greet>")
						extracted_seqs[-1][uter_split.index(greetings[greet_i])] = "<politeness_buyer_greet>"
						tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + greetings[greet_i], " <politeness_buyer_greet> ")
				else:
					if greetings[greet_i] in uter:
						politeness_buyer_greetings += 1
						recommendation_data_uter[recommendation_feature_mapping["politeness_buyer_greet"]] = 1
						tmp_strategy_sequences.append("<politeness_buyer_greet>")
						fine_intents[-1].append("<politeness_buyer_greet>")
						extracted_seqs[-1][uter_split.index(greetings[greet_i].split()[0])] = "<politeness_buyer_greet>"
						tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + greetings[greet_i], " <politeness_buyer_greet> ")

			for grad_i in range(len(gratitute)):
				if gratitute[grad_i] in word_tokenized:
					politeness_buyer_gratitude += 1
					recommendation_data_uter[recommendation_feature_mapping["politeness_buyer_gratitude"]] = 1
					tmp_strategy_sequences.append("<politeness_buyer_gratitude>")
					fine_intents[-1].append("<politeness_buyer_gratitude>")
					extracted_seqs[-1][uter_split.index(gratitute[grad_i])] = "<politeness_buyer_gratitude>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + gratitute[grad_i], " <politeness_buyer_gratitude> ")

			please_index = -1
			for word_i in range(len(word_tokenized)):
				if word_tokenized[word_i] == "please" or word_tokenized[word_i] == "pls":
					please_index = word_i
					break
			if please_index != -1:
				if word_tokenized[please_index-1] != ">" and word_tokenized[please_index-1] != "." and word_tokenized[please_index-1] != "?" and word_tokenized[please_index-1] != "!":
					politeness_buyer_please += 1
					recommendation_data_uter[recommendation_feature_mapping["politeness_buyer_please"]] = 1
					tmp_strategy_sequences.append("<politeness_buyer_please>")
					fine_intents[-1].append("<politeness_buyer_please>")
					extracted_seqs[-1][please_index] = "<politeness_buyer_please>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" please", " <politeness_buyer_please> ").replace("pls", "<politeness_buyer_please>")
				else:
					politeness_buyer_please_s += 1
					recommendation_data_uter[recommendation_feature_mapping["politeness_buyer_please_s"]] = 1
					tmp_strategy_sequences.append("<politeness_buyer_please_s>")
					fine_intents[-1].append("<politeness_buyer_please_s>")
					extracted_seqs[-1][please_index] = "<politeness_buyer_please_s>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" please", " <politeness_buyer_please_s> ").replace("pls", "<politeness_buyer_please_s>")


			for word_i, word in enumerate(word_tokenized):
				word = lemmatizer.lemmatize(word)
				if word in liwc_friend and (previous_word != "your" and previous_word != "ur"):
					social_distance_buyer += 1.0
					social_distance_count_buyer += 1.0
				if word in liwc_family and (previous_word != "your" and previous_word != "ur"):
					social_distance_buyer += 0.0
					social_distance_count_buyer += 1.0
				if word in factive:
					factive_count_buyer += 1
					recommendation_data_uter[recommendation_feature_mapping["factive_count_buyer"]] = 1
					tmp_strategy_sequences.append("<factive_count_buyer>")
					fine_intents[-1].append("<factive_count_buyer>")
					extracted_seqs[-1][word_i] = "<factive_count_buyer>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <factive_count_buyer> ")

				if word in first_person_singular:
					first_person_singular_count_buyer += 1
					recommendation_data_uter[recommendation_feature_mapping["first_person_singular_count_buyer"]] = 1
					tmp_strategy_sequences.append("<first_person_singular_count_buyer>")
					fine_intents[-1].append("<first_person_singular_count_buyer>")
					extracted_seqs[-1][word_i] = "<first_person_singular_count_buyer>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word +" ", " <first_person_singular_count_buyer> ")
				elif word in first_person_plural:
					first_person_plural_count_buyer += 1
					recommendation_data_uter[recommendation_feature_mapping["first_person_plural_count_buyer"]] = 1
					tmp_strategy_sequences.append("<first_person_plural_count_buyer>")
					fine_intents[-1].append("<first_person_plural_count_buyer>")
					extracted_seqs[-1][word_i] = "<first_person_plural_count_buyer>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <first_person_plural_count_buyer> ")
				elif word in third_person_plural:
					third_person_plural_buyer += 1
					recommendation_data_uter[recommendation_feature_mapping["third_person_plural_buyer"]] = 1
					tmp_strategy_sequences.append("<third_person_plural_buyer>")
					fine_intents[-1].append("<third_person_plural_buyer>")
					extracted_seqs[-1][word_i] = "<third_person_plural_buyer>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <third_person_plural_buyer> ")
				elif word in third_person_singular:
					third_person_singular_buyer += 1
					recommendation_data_uter[recommendation_feature_mapping["third_person_singular_buyer"]] = 1
					tmp_strategy_sequences.append("<third_person_singular_buyer>")
					fine_intents[-1].append("<third_person_singular_buyer>")
					extracted_seqs[-1][word_i] = "<third_person_singular_buyer>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <third_person_singular_buyer> ")
				if word in assertive:
					assertive_count_buyer += 1
					recommendation_data_uter[recommendation_feature_mapping["assertive_count_buyer"]] = 1
					tmp_strategy_sequences.append("<assertive_count_buyer>")
					fine_intents[-1].append("<assertive_count_buyer>")
					extracted_seqs[-1][word_i] = "<assertive_count_buyer>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <assertive_count_buyer> ")
				if word in hedges_word:
					if _first_price and first_price:
						example_arousal_tmp.append(word + "," + "N/A" + "," + uter.replace(",", ";"))
					hedge_count_buyer += 1
					recommendation_data_uter[recommendation_feature_mapping["hedge_count_buyer"]] = 1
					tmp_strategy_sequences.append("<hedge_count_buyer>")
					fine_intents[-1].append("<hedge_count_buyer>")
					extracted_seqs[-1][word_i] = "<hedge_count_buyer>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <hedge_count_buyer> ")
				if word in pos:
					buyer_pos_sentiment += 1
					recommendation_data_uter[recommendation_feature_mapping["buyer_pos_sentiment"]] = 1
					tmp_strategy_sequences.append("<buyer_pos_sentiment>")
					fine_intents[-1].append("<buyer_pos_sentiment>")
					extracted_seqs[-1][word_i] = "<buyer_pos_sentiment>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <buyer_pos_sentiment> ")
				if word in neg:
					buyer_neg_sentiment += 1
					recommendation_data_uter[recommendation_feature_mapping["buyer_neg_sentiment"]] = 1
					tmp_strategy_sequences.append("<buyer_neg_sentiment>")
					fine_intents[-1].append("<buyer_neg_sentiment>")
					extracted_seqs[-1][word_i] = "<buyer_neg_sentiment>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <buyer_neg_sentiment> ")
				if word in dominance:
					dominance_count_buyer += 1
					dominance_avg_buyer += dominance[word]
					valence_avg_buyer += valence[word]
					arousal_avg_buyer += arousal[word]
				total_words_buyer += 1
				stat_tmp.append(word)
				vocab_tmp.append(word)
				previous_word = word
			total_uterance_buyer += 1
			recommendation_data_uter_cumu = [a + b for a, b in zip(recommendation_data_uter_cumu, recommendation_data_uter[:-2])]
			recommendation_raw_utterance_tmp.append(tmp_strategies_embedding_text)
			strategy_sequences.append(tmp_strategy_sequences)

		if "<selle>" in uter and "<offer " not in uter and "<accept>" not in uter:
			variance_examples.append(uter)
			for key in variance_examples_labels:
				variance_examples_labels[key].append(0)

			#tmp_strategy_sequences.append("<selle>")
			if "throw in" in uter or "throwing in" in uter:
				seller_trade_in = 1
				recommendation_data_uter[recommendation_feature_mapping["seller_trade_in"]] = 1
				tmp_strategy_sequences.append("<seller_trade_in>")
				fine_intents[-1].append("<seller_trade_in>")
				if "throw" in uter_split:
					extracted_seqs[-1][uter_split.index("throw")] = "<seller_trade_in>"
				else:
					extracted_seqs[-1][uter_split.index("throwing")] = "<seller_trade_in>"
				tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace("throw in", "<seller_trade_in>").replace("throwing in", "<seller_trade_in>")
				if first_price and _first_price:
					tmp_strategies[-1].append("<1_Trade_In>")
				else:
					tmp_strategies[-1].append("<2_Trade_In>")
				variance_examples_labels["seller_trade_in"][-1] = 1
			if "deliver" in uter:
				recommendation_data_uter[recommendation_feature_mapping["seller_trade_in"]] = 1
				fine_intents[-1].append("<seller_trade_in>")
				if "deliver" in uter_split:
					extracted_seqs[-1][uter_split.index("deliver")] = "<seller_trade_in>"
				elif "delivery" in uter_split:
					extracted_seqs[-1][uter_split.index("delivery")] = "<seller_trade_in>"

				seller_deliver += 1

			if len(re.findall(r"\d+", uter)) > 0:
				#seller_propose_visit = True
				for possible_price in re.findall(r"\d+", uter):
					if 1 > float(possible_price)/float(price) > 0.7  and abs(float(possible_price) - float(price)) < abs(float(buyer_first_price) - float(price)):
						if seller_propose_visit:
							seller_propose += 1
							tmp_strategy_sequences.append("<seller_propose>")
							fine_intents[-1].append("<seller_propose>")
							if str(possible_price) in uter_split:
								extracted_seqs[-1][uter_split.index(str(possible_price))] = "<seller_propose>"
							recommendation_data_uter[recommendation_feature_mapping["seller_propose"]] = 1
							tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(possible_price, "<seller_propose>")
							tmp_strategies[-1].append("<Propose_New_Price>")
							seller_propose_visit = False
							variance_examples_labels["seller_propose"][-1] = 1
						if _first_price:
							seller_first_price = float(possible_price)
					if 1 > float(possible_price)/float(price) > 0.5:
						if who_propose_visit:
							who_propose = 1
							recommendation_data_uter[recommendation_feature_mapping["who_propose"]] = 1
							who_propose_visit = False
							variance_examples_labels["who_propose"][-1] = 1
				_first_price = False
				if seller_first_price == 0.0:
					_first_price = True


			#uterrance wise analysis
			## TODO (INSERT CODE HERE FOR CLASSIFIER BASED STRATEGIES) # 530 in regresssion.py
			for greet_i in range(len(greetings)):
				if greet_i <= 7:
					if greetings[greet_i] in word_tokenized:
						politeness_seller_greetings += 1
						recommendation_data_uter[recommendation_feature_mapping["politeness_seller_greet"]] = 1
						tmp_strategy_sequences.append("<politeness_seller_greet>")
						fine_intents[-1].append("<politeness_seller_greet>")
						extracted_seqs[-1][uter_split.index(greetings[greet_i])] = "<politeness_seller_greet>"
						tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + greetings[greet_i], " <politeness_seller_greet> ")
						if first_price and _first_price:
							tmp_strategies[-1].append("<1_Greetings>")
						else:
							tmp_strategies[-1].append("<2_Greetings>")
						variance_examples_labels["politeness_seller_greet"][-1] = 1
				else:
					if greetings[greet_i] in uter:
						politeness_seller_greetings += 1
						recommendation_data_uter[recommendation_feature_mapping["politeness_seller_greet"]] = 1
						tmp_strategy_sequences.append("<politeness_seller_greet>")
						fine_intents[-1].append("<politeness_seller_greet>")
						extracted_seqs[-1][uter_split.index(greetings[greet_i].split()[0])] = "<politeness_seller_greet>"
						tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + greetings[greet_i], " <politeness_seller_greet> ")
						if first_price and _first_price:
							tmp_strategies[-1].append("<1_Greetings>")
						else:
							tmp_strategies[-1].append("<2_Greetings>")
						variance_examples_labels["politeness_seller_greet"][-1] = 1
			
			for sorry_i in range(len(apology)):
				if sorry_i <= 1:
					if apology[sorry_i] in word_tokenized:
						politeness_seller_apology += 1
						if first_price and _first_price:
							tmp_strategies[-1].append("<1_Apology>")
						else:
							tmp_strategies[-1].append("<2_Apology>")
				else:
					if apology[sorry_i] in uter:
						politeness_seller_apology += 1
						if first_price and _first_price:
							tmp_strategies[-1].append("<1_Apology>")
						else:
							tmp_strategies[-1].append("<2_Apology>")
					
				
			for grad_i in range(len(gratitute)):
				if gratitute[grad_i] in word_tokenized:
					politeness_seller_gratitude += 1
					recommendation_data_uter[recommendation_feature_mapping["politeness_seller_gratitude"]] = 1
					tmp_strategy_sequences.append("<politeness_seller_gratitude>")
					fine_intents[-1].append("<politeness_seller_gratitude>")
					extracted_seqs[-1][uter_split.index(gratitute[grad_i])] = "<politeness_seller_gratitude>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + gratitute[grad_i], " <politeness_seller_gratitude> ")
					if first_price and _first_price:
						tmp_strategies[-1].append("<1_gratitude>")
					else:
						tmp_strategies[-1].append("<2_gratitude>")
					variance_examples_labels["politeness_seller_gratitude"][-1] = 1

			please_index = -1
			for word_i in range(len(word_tokenized)):
				if word_tokenized[word_i] == "please" or word_tokenized[word_i] == "pls":
					please_index = word_i
					break
			if please_index != -1:
				if word_tokenized[please_index-1] != ">" and word_tokenized[please_index-1] != "." and word_tokenized[please_index-1] != "?" and word_tokenized[please_index-1] != "!":
					politeness_seller_please += 1
					recommendation_data_uter[recommendation_feature_mapping["politeness_seller_please"]] = 1
					tmp_strategy_sequences.append("<politeness_seller_please>")
					fine_intents[-1].append("<politeness_seller_please>")
					extracted_seqs[-1][please_index] = "<politeness_seller_please>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace("please", "<politeness_seller_please>").replace("pls", "<politeness_seller_please>")
					if first_price and _first_price:
						tmp_strategies[-1].append("<1_Please>")
					else:
						tmp_strategies[-1].append("<2_Please>")
					variance_examples_labels["politeness_seller_please"][-1] = 1
				else:
					politeness_seller_please_s += 1
					recommendation_data_uter[recommendation_feature_mapping["politeness_seller_please_s"]] = 1
					tmp_strategy_sequences.append("<politeness_seller_please_s>")
					fine_intents[-1].append("<politeness_seller_please_s>")
					extracted_seqs[-1][please_index] = "<politeness_seller_please_s>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace("please", "<politeness_seller_please_s>").replace("pls", "<politeness_seller_please_s>")
					if first_price and _first_price:
						tmp_strategies[-1].append("<1_Please_Start>")
					else:
						tmp_strategies[-1].append("<2_Please_Start>")
					variance_examples_labels["politeness_seller_please_s"][-1] = 1

			previous_word = ""
			for word_i, word in enumerate(word_tokenized):
				word = lemmatizer.lemmatize(word)
				if word in liwc_informal_dic and word != "ha" and word != "yes" and word != "like" and word != "absolutely" and word != "agree" and word != "ok":
					if word != "well":
						liwc_informal += 1
						recommendation_data_uter[recommendation_feature_mapping["liwc_informal"]] = 1
						tmp_strategy_sequences.append("<liwc_informal>")
						fine_intents[-1].append("<liwc_informal>")
						extracted_seqs[-1][word_i] = "<liwc_informal>"
						tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <liwc_informal> ")
						if first_price and _first_price:
							tmp_strategies[-1].append("<1_Informal_Word>")
						else:
							tmp_strategies[-1].append("<2_Informal_Word>")
						variance_examples_labels["liwc_informal"][-1] = 1
					else:
						if previous_word == ">":
							if first_price and _first_price:
								tmp_strategies[-1].append("<1_Informal_Word>")
							else:
								tmp_strategies[-1].append("<2_Informal_Word>")
							liwc_informal += 1
							recommendation_data_uter[recommendation_feature_mapping["liwc_informal"]] = 1
							tmp_strategy_sequences.append("<liwc_informal>")
							fine_intents[-1].append("<liwc_informal>")
							extracted_seqs[-1][word_i] = "<liwc_informal>"
							tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <liwc_informal> ")
							variance_examples_labels["liwc_informal"][-1] = 1
				if word in liwc_certain and word != "certain":
					if word == "sure":
						if previous_word != "not" and previous_word != ">" and "?" not in uter:
							liwc_certainty += 1
							recommendation_data_uter[recommendation_feature_mapping["liwc_certainty"]] = 1
							tmp_strategy_sequences.append("<liwc_certainty>")
							fine_intents[-1].append("<liwc_certainty>")
							extracted_seqs[-1][word_i] = "<liwc_certainty>"
							tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <liwc_certainty> ")
							if first_price and _first_price:
								tmp_strategies[-1].append("<1_Certain_Word>")
							else:
								tmp_strategies[-1].append("<2_Certain_Word>")
							variance_examples_labels["liwc_certainty"][-1] = 1
							# if not (first_price and _first_price):
							# 	print word + "," + uter.replace(",","")
					else:
						liwc_certainty += 1
						tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <liwc_certainty> ")
						recommendation_data_uter[recommendation_feature_mapping["liwc_certainty"]] = 1
						tmp_strategy_sequences.append("<liwc_certainty>")
						fine_intents[-1].append("<liwc_certainty>")
						extracted_seqs[-1][word_i] = "<liwc_certainty>"
						if first_price and _first_price:
							tmp_strategies[-1].append("<1_Certain_Word>")
						else:
							tmp_strategies[-1].append("<2_Certain_Word>")
						variance_examples_labels["liwc_certainty"][-1] = 1
						# if not (first_price and _first_price):
						# 	print word + "," + uter.replace(",","")
				if word in liwc_friend and (previous_word != "your" and previous_word != "ur") and not word.startswith("bud"):
					social_distance_seller += 1.0
					social_distance_count += 1.0
					recommendation_data_uter[recommendation_feature_mapping["friend"]] = 1
					tmp_strategy_sequences.append("<friend>")
					fine_intents[-1].append("<friend>")
					extracted_seqs[-1][word_i] = "<friend>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <friend> ")
					if first_price and _first_price:
						tmp_strategies[-1].append("<1_Friend_Word>")
					else:
						tmp_strategies[-1].append("<2_Friend_Word>")
					variance_examples_labels["friend"][-1] = 1
				if word in liwc_family and (previous_word != "your" and previous_word != "ur"):
					if word == "family":
						if previous_word == "my":
							social_distance_seller += 0.0
							recommendation_data_uter[recommendation_feature_mapping["family"]] = 1
							tmp_strategy_sequences.append("<family>")
							fine_intents[-1].append("<family>")
							extracted_seqs[-1][word_i] = "<family>"
							tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <family> ")
							social_distance_count += 1.0
							if first_price and _first_price:
								tmp_strategies[-1].append("<1_Family>")
							else:
								tmp_strategies[-1].append("<2_Family>")
							variance_examples_labels["family"][-1] = 1
					else:
						social_distance_seller += 0.0
						recommendation_data_uter[recommendation_feature_mapping["family"]] = 1
						tmp_strategy_sequences.append("<family>")
						fine_intents[-1].append("<family>")
						extracted_seqs[-1][word_i] = "<family>"
						tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <family> ")
						social_distance_count += 1.0
						if first_price and _first_price:
							tmp_strategies[-1].append("<1_Family>")
						else:
							tmp_strategies[-1].append("<2_Family>")
						variance_examples_labels["family"][-1] = 1
				if word in liwc_personal_concern:
					personal_concern_seller += 1
					recommendation_data_uter[recommendation_feature_mapping["personal_concern_seller"]] = 1
					tmp_strategy_sequences.append("<personal_concern_seller>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <personal_concern_seller> ")
				if word in factive:
					keywords["factive_count_seller"] = word
					factive_count_seller += 1
					recommendation_data_uter[recommendation_feature_mapping["factive_count_seller"]] = 1
					tmp_strategy_sequences.append("<factive_count_seller>")
					fine_intents[-1].append("<factive_count_seller>")
					extracted_seqs[-1][word_i] = "<factive_count_seller>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <factive_count_seller> ")
					variance_examples_labels["factive_count_seller"][-1] = 1
				if word in first_person_singular:
					first_person_singular_count_seller += 1
					recommendation_data_uter[recommendation_feature_mapping["first_person_singular_count_seller"]] = 1
					tmp_strategy_sequences.append("<first_person_singular_count_seller>")
					fine_intents[-1].append("<first_person_singular_count_seller>")
					extracted_seqs[-1][word_i] = "<first_person_singular_count_seller>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word +" ", " <first_person_singular_count_seller> ")
					variance_examples_labels["first_person_singular_count_seller"][-1] = 1
				elif word in first_person_plural:
					first_person_plural_count_seller += 1
					recommendation_data_uter[recommendation_feature_mapping["first_person_plural_count_seller"]] = 1
					tmp_strategy_sequences.append("<first_person_plural_count_seller>")
					fine_intents[-1].append("<first_person_plural_count_seller>")
					extracted_seqs[-1][word_i] = "<first_person_plural_count_seller>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <first_person_plural_count_seller> ")
					variance_examples_labels["first_person_plural_count_seller"][-1] = 1
				elif word in third_person_plural:
					third_person_plural_seller += 1
					recommendation_data_uter[recommendation_feature_mapping["third_person_plural_seller"]] = 1
					tmp_strategy_sequences.append("<third_person_plural_seller>")
					fine_intents[-1].append("<third_person_plural_seller>")
					extracted_seqs[-1][word_i] = "<third_person_plural_seller>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <third_person_plural_seller> ")
					variance_examples_labels["third_person_plural_seller"][-1] = 1
				elif word in third_person_singular:
					third_person_singular_seller += 1
					recommendation_data_uter[recommendation_feature_mapping["third_person_singular_seller"]] = 1
					tmp_strategy_sequences.append("<third_person_singular_seller>")
					fine_intents[-1].append("<third_person_singular_seller>")
					extracted_seqs[-1][word_i] = "<third_person_singular_seller>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <third_person_singular_seller> ")
					variance_examples_labels["third_person_singular_seller"][-1] = 1
				if word in dominance:
					dominance_count_seller += 1
					dominance_avg_seller += dominance[word]
					valence_avg_seller += valence[word]
					arousal_avg_seller += arousal[word]
				if word in assertive:
					keywords["assertive_count_seller"] = word
					assertive_count_seller += 1
					recommendation_data_uter[recommendation_feature_mapping["assertive_count_seller"]] = 1
					tmp_strategy_sequences.append("<assertive_count_seller>")
					fine_intents[-1].append("<assertive_count_seller>")
					extracted_seqs[-1][word_i] = "<assertive_count_seller>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <assertive_count_seller> ")
				if word in hedges_word:
					#print word + "," + uter.replace(",","")
					keywords["hedge_count_seller"] = word
					recommendation_data_uter[recommendation_feature_mapping["hedge_count_seller"]] = 1
					tmp_strategy_sequences.append("<hedge_count_seller>")
					fine_intents[-1].append("<hedge_count_seller>")
					extracted_seqs[-1][word_i] = "<hedge_count_seller>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <hedge_count_seller> ")
					hedge_count_seller += 1
					if o_propose_visit:
						propose_hedge_tmp += 1
					variance_examples_labels["hedge_count_seller"][-1] = 1
				if word in pos:
					seller_pos_sentiment += 1
					recommendation_data_uter[recommendation_feature_mapping["seller_pos_sentiment"]] = 1
					tmp_strategy_sequences.append("<seller_pos_sentiment>")
					fine_intents[-1].append("<seller_pos_sentiment>")
					extracted_seqs[-1][word_i] = "<seller_pos_sentiment>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <seller_pos_sentiment> ")
					variance_examples_labels["seller_pos_sentiment"][-1] = 1
				if word in neg:
					seller_neg_sentiment += 1
					recommendation_data_uter[recommendation_feature_mapping["seller_neg_sentiment"]] = 1
					tmp_strategy_sequences.append("<seller_neg_sentiment>")
					fine_intents[-1].append("<seller_neg_sentiment>")
					extracted_seqs[-1][word_i] = "<seller_neg_sentiment>"
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <seller_neg_sentiment> ")
					variance_examples_labels["seller_neg_sentiment"][-1] = 1
				if word in lexicon_diff_dic_pos:
					number_of_diff_dic_pos += 1
					recommendation_data_uter[recommendation_feature_mapping["number_of_diff_dic_pos"]] = 1
					tmp_strategy_sequences.append("<number_of_diff_dic_pos>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <number_of_diff_dic_pos> ")
				if word in lexicon_diff_dic_neg:
					number_of_diff_dic_neg += 1
					recommendation_data_uter[recommendation_feature_mapping["number_of_diff_dic_neg"]] = 1
					tmp_strategy_sequences.append("<number_of_diff_dic_neg>")
					tmp_strategies_embedding_text = tmp_strategies_embedding_text.replace(" " + word, " <number_of_diff_dic_neg> ")
				total_words_seller += 1
				stat_tmp.append(word)
				vocab_tmp.append(word)
				previous_word = word 
			total_uterance_seller += 1
			recommendation_data_uter_cumu = [a + b for a, b in zip(recommendation_data_uter_cumu, recommendation_data_uter[:-2])]
			recommendation_raw_utterance_tmp.append(tmp_strategies_embedding_text)
			strategy_sequences.append(tmp_strategy_sequences)

			if len(strategy_sequences) > 1:
				if ",".join(strategy_sequences[-2]) not in majority_rules:
					majority_rules[",".join(strategy_sequences[-2])] = Counter()
					majority_rules[",".join(strategy_sequences[-2])][",".join(strategy_sequences[-1])]+=1
				else:
					majority_rules[",".join(strategy_sequences[-2])][",".join(strategy_sequences[-1])]+=1

			recommendation_template_tmp = [1,0] + recommendation_data_uter[:-2] 
			key = "".join([str(int(a)) for a in recommendation_template_tmp])
			if key not in recommendation_template and keywords:
				recommendation_template[key] = [uter, keywords]


		uter_index += 1
		extracted_seqs[-1] = extracted_seqs[-1][1:]
		
		bag_of_strategies.append(recommendation_data_uter)							 
		strategy_embedding_text_dialog += previous_strategies_embedding + " " + tmp_strategies_embedding_text + "\n"
		fine_intents[-1].append("<end>")

		# if "<offer " in uter:
		# 	final = re.findall(r'\d+', uter)[0]

		if (not (first_price and _first_price) and portion_index == 1) or (uter_index == number_of_uter): # #uter_index >= portion_index*number_of_uter/4.0 and portion_index <= 4:
			if len(tmp) == 0 and uter_index == number_of_uter:
				tmp = [0.0]*feature_size
				tmp[-10] = who_propose

			# if portion_index == 1:
			# 	stage_1_stat += vocab_tmp
			# 	vocab_tmp = list()
			# else:
			# 	stage_2_stat += vocab_tmp

			portion_index += 1
			tmp += tmp_complex

			#sentiment features
			tmp.append(seller_neg_sentiment)
			tmp.append(seller_pos_sentiment)
			tmp.append(buyer_neg_sentiment)
			tmp.append(buyer_pos_sentiment)

			#stubborn features
			if (dominance_count_buyer != 0):
				tmp.append(dominance_avg_buyer/dominance_count_buyer)
				tmp.append(arousal_avg_buyer/dominance_count_buyer)
			else:
				tmp.append(0)
				tmp.append(0)
			if (dominance_count_seller != 0):
				tmp.append(dominance_avg_seller/dominance_count_seller)
				tmp.append(arousal_avg_seller/dominance_count_seller)
			else:
				tmp.append(0)
				tmp.append(0)

			tmp.append(first_person_plural_count_seller)
			tmp.append(first_person_singular_count_seller)
			tmp.append(first_person_plural_count_buyer)
			tmp.append(first_person_singular_count_buyer)
			tmp.append(third_person_singular_seller)
			tmp.append(third_person_plural_seller)
			tmp.append(third_person_singular_buyer)
			tmp.append(third_person_plural_buyer)
			tmp.append(number_of_diff_dic_pos)
			tmp.append(number_of_diff_dic_neg)

			#most informative ones	
			if total_uterance_seller == 0:
				tmp.append(0)
			else:
				tmp.append(total_words_seller/total_uterance_seller)
			if total_uterance_buyer == 0:
				tmp.append(0)
			else:
				tmp.append(total_words_buyer/total_uterance_buyer)
			tmp.append(buyer_propose)
			tmp.append(seller_propose)


			tmp.append(float(buyer_first_price))
			tmp.append(float(seller_first_price))
			tmp.append(float(price))
			tmp.append((float(buyer_first_price) - float(price))/float(price))

			tmp.append(hedge_count_seller)
			tmp.append(hedge_count_buyer)
			tmp.append(assertive_count_seller)
			tmp.append(assertive_count_buyer)
			tmp.append(factive_count_seller)
			tmp.append(factive_count_buyer)

			
			tmp.append(who_propose)
			tmp.append(seller_trade_in)
			tmp.append(personal_concern_seller)
			tmp.append(sg_concern)
			if social_distance_count != 0:
				tmp.append(social_distance_seller/social_distance_count)
			else:
				tmp.append(2.0)
			tmp.append(liwc_certainty)
			tmp.append(liwc_informal)
			#tmp.append(politeness_seller_apology)
			#tmp.append(politeness_seller_greetings)
			tmp.append(politeness_seller_please)
			tmp.append(politeness_seller_gratitude)
			tmp.append(politeness_seller_please_s)

			if uter_index != number_of_uter:
				tmp_complex = [0,0,0,0,0]
				seller_neg_sentiment = 0.0
				seller_pos_sentiment = 0.0
				buyer_neg_sentiment = 0.0
				buyer_pos_sentiment = 0.0
				dominance_avg_buyer = 0.0
				dominance_count_buyer = 0.0
				arousal_avg_buyer = 0.0
				dominance_count_buyer = 0.0
				dominance_avg_seller = 0.0
				dominance_count_seller = 0.0
				arousal_avg_seller = 0.0
				dominance_count_seller = 0.0
				first_person_plural_count_seller = 0.0
				first_person_singular_count_seller = 0.0
				first_person_plural_count_buyer = 0.0
				first_person_singular_count_buyer = 0.0
				third_person_singular_seller = 0.0
				third_person_plural_seller = 0.0
				third_person_singular_buyer = 0.0
				third_person_plural_buyer = 0.0
				number_of_diff_dic_pos = 0.0
				number_of_diff_dic_neg = 0.0
				total_uterance_seller  = 0.0
				total_words_seller  = 0.0
				total_words_buyer = 0.0
				total_uterance_buyer = 0.0
				buyer_propose  = 0.0
				seller_propose  = 0.0
				hedge_count_seller = 0.0
				hedge_count_buyer = 0.0
				assertive_count_seller = 0.0
				assertive_count_buyer = 0.0
				factive_count_seller = 0.0
				factive_count_buyer = 0.0
				seller_trade_in = 0
				personal_concern_seller = 0
				sg_concern = 0
				social_distance_seller = 0.0
				social_distance_count  = 0.0
				liwc_certainty = 0.0
				liwc_informal = 0.0
				#politeness_seller_apology = 0.0
				#politeness_seller_greetings = 0.0
				politeness_seller_please = 0.0
				politeness_seller_gratitude = 0.0
				politeness_seller_please_s = 0.0

		previous = uter
		uter_index_overall += 1

	return fine_intents, bag_of_strategies, extracted_seqs


def calculate_ngram_features(ngram_dic, features, uter):
	tmp = list()
	for i in word_tokenize(uter):
		if i not in stopWords:
			tmp.append(lemmatizer.lemmatize(i))
	for word in word_grams(tmp):
		if word in ngram_dic:
			features[ngram_dic[word]] += 1
	return features



def word_grams(words, min=1, max=4):
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s


if __name__ == "__main__": main()
