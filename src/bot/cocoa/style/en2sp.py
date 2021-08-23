# -*- coding: utf-8 -*-

import random
import spacy
from nltk.tokenize import wordpunct_tokenize

""" Convert English to Spanglish! """

__author__ = 'eahn1'

nlp_en = spacy.load('en')
nlp_sp = spacy.load('es')

def read_tables(fname):
	# with open("data/chat_prev/en_rule.txt") as f:
	# 	en_orig = [line.replace('\n','') for line in f.readlines()]
	# with open("data/chat_prev/sp_rule.txt") as f:
	# 	sp_orig = [line.replace('\n','') for line in f.readlines()]

	with open(fname) as f:
		en_orig = [line.split(' ||| ')[0] for line in f.readlines()]
		sp_orig = [line.split(' ||| ')[1].replace('\n','') for line in f.readlines()]

	return dict([('en', en_orig), ('sp', sp_orig)])

tables_dct = read_tables("style/en-sp_rule.txt")

def read_fastalign(align_fname):
	with open(align_fname) as f:
		pass

# idx&txt params are redundant but to-be-modified later
def sp_matrix_eng_nouns(en_txt, idx): #tables_dct, 
	""" en_txt: string of words (not list)
		tables_dct: contains 2 tables, keys = en,sp
		idx: int of which index in table is this sentence
		[return] string of words (list separated by space)
	"""
	en_doc = nlp_en(unicode(en_txt, "utf-8"))
	en_nouns = [en_token.text for en_token in en_doc if en_token.pos_ in ['NOUN', 'PROPN']]
	# avoid "who"
	en_nouns = [noun for noun in en_nouns if not noun==u"who"]
	# print "EN NOUNS", en_nouns

	sp_txt = tables_dct['sp'][idx]
	sp_doc = nlp_sp(unicode(sp_txt, "utf-8"))
	sp_nouns = []
	for orig_i, sp_token in enumerate(sp_doc):
		if sp_token.pos_ in ['NOUN', 'PROPN']: sp_nouns.append((sp_token.text, orig_i))
		# capture infinitive verbs ==> turn into noun (e.g. nadar = swimming)
		if sp_token.pos_=='VERB' and sp_token.tag_.endswith('Inf'): sp_nouns.append((sp_token.text, orig_i))
		
	# print "SP NOUNS", sp_nouns

	cm_txt = [sp_token.text for sp_token in sp_doc]
	for noun_idx, (sp_token, orig_i) in enumerate(sp_nouns):
		if not en_nouns: break
		if noun_idx < len(en_nouns)-1: #valid, or else is last
			cm_txt[orig_i] = en_nouns[noun_idx]
		else: # get last noun (probably last one anyway??)
			cm_txt[orig_i] = en_nouns[-1]
	return " ".join(cm_txt)

def print_lex():
	print "STYLE 1: LEXICAL on Rule-based Bot // 6 Feb 2018"
	print "KEY:\n1. Spanish (orig)\n2. English (orig)\n3. CodeMix"
	print "*"*10

	for i, en_sent in enumerate(tables_dct['en']):
		print tables_dct['sp'][i]
		print en_sent
		print sp_matrix_eng_nouns(en_sent, i).encode('utf-8') # encoding req'd to save print logs
		print "*"*10
		# if i==10: break

def create_synt_cm(en_txt, sp_txt, start_en_bool):
	
	# en_txt = tables_dct['en'][idx]
	en_doc = nlp_en(unicode(en_txt, "utf-8"))
	# sp_txt = tables_dct['sp'][idx]
	sp_doc = nlp_sp(unicode(sp_txt, "utf-8"))

	if start_en_bool:
		pass

	# TO BE CONTINUED



# TEST/PLAY AROUND



sent_sp = "Tengo dos amigos que estudiaron estudios islámicos."
sent_en = "I have two friends who studied islamic studies."

start_en_bool = random.choice([True, False])
create_synt_cm(sent_en, sent_sp, start_en_bool)

# test_dct = dict([('en', [sent_en]), ('sp', [sent_sp])])
# print sp_matrix_eng_nouns(sent_en, test_dct, 0)

# print_lex()

# visualize POS for first n examples
'''
n = 50
for i, en_txt in enumerate(tables_dct['en'][:n]):
	sp_txt = tables_dct['sp'][i]
	en_doc = nlp_en(unicode(en_txt, "utf-8"))
	sp_doc = nlp_sp(unicode(sp_txt, "utf-8"))
	print en_txt
	print [en_token.pos_ for en_token in en_doc]
	print sp_txt
	print [sp_token.pos_ for sp_token in sp_doc]
	print "*"*10
'''





# possible stuff in spacy's doc object:
# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)  