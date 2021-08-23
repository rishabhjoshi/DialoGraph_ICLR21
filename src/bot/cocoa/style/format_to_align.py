# -*- coding: utf-8 -*-
import sys
import spacy
from nltk.tokenize import wordpunct_tokenize
import codecs

""" tokenize and get into parallel text format to feed into fast_align.
	both texts get tokenized.
	file1 = ENGLISH, file2 = SPANISH, file3 = OUTFILE
"""

__author__ = 'eahn1'

left_txt_file = sys.argv[1] # ENGLISH
right_txt_file = sys.argv[2] # SPANISH
outfile = sys.argv[3]

newfile = []
nlp_en = spacy.load('en')
nlp_sp = spacy.load('es')

# with open(right_txt_file, 'r') as f:
# 	right_txt = f.readlines()

with codecs.open(right_txt_file, "r",encoding='utf-8') as f:
	right_txt = [row.lower() for row in f]

with open(left_txt_file, 'r') as f:
	left_txt = [row.replace('\n','').lower() for row in f.readlines()]

for i, left_line in enumerate(left_txt):
	en_doc = nlp_en(unicode(left_line, "utf-8"))
	sp_doc = nlp_sp(unicode(right_txt[i]), "utf-8") #encode("utf-8")
	new_left = " ".join([en_token.text for en_token in en_doc]).encode("utf-8")
	new_right = " ".join([sp_token.text for sp_token in sp_doc]).encode("utf-8")
	# new_left = " ".join(wordpunct_tokenize(left_line))
	# new_right = " ".join(wordpunct_tokenize(right_txt[i]))
	new_line = "{} ||| {}".format(new_left, new_right)
	newfile.append(new_line)

with open(outfile, 'w') as w:
	for line in newfile:
		w.write(line)