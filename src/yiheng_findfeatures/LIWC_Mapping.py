import json 

import os
curr_file_path = os.path.dirname(os.path.abspath(__file__)) + '/'

def sentiment():
	#first we want to get a list of LIWC
	index = 0
	neg = list()
	pos = list()

	liwc = open(curr_file_path + 'LIWC2007_English080730.dic').read().split('\n')
	boo = False

	for line in liwc:
		if line == "%":
			boo = not boo
			continue
		if boo:
			continue
		else:
			tmp = line.split('\t')
			for i in range(1, len(tmp)):
				if tmp[i] == '126':
					pos.append(tmp[0])
				if tmp[i] == '127':
					neg.append(tmp[0])
	return pos, neg

def family():
	#first we want to get a list of LIWC
	index = 0
	pos = list()

	liwc = open(curr_file_path + 'LIWC2007_English080730.dic').read().split('\n')
	boo = False

	for line in liwc:
		if line == "%":
			boo = not boo
			continue
		if boo:
			continue
		else:
			tmp = line.split('\t')
			for i in range(1, len(tmp)):
				if tmp[i] == '122':
					pos.append(tmp[0])
	return pos

def friend():
	#first we want to get a list of LIWC
	index = 0
	pos = list()

	liwc = open(curr_file_path + 'LIWC2007_English080730.dic').read().split('\n')
	boo = False

	for line in liwc:
		if line == "%":
			boo = not boo
			continue
		if boo:
			continue
		else:
			tmp = line.split('\t')
			for i in range(1, len(tmp)):
				if tmp[i] == '123':
					pos.append(tmp[0])
	return pos


def i():
	#first we want to get a list of LIWC
	index = 0
	pos = list()

	liwc = open(curr_file_path + 'LIWC2007_English080730.dic').read().split('\n')
	boo = False

	for line in liwc:
		if line == "%":
			boo = not boo
			continue
		if boo:
			continue
		else:
			tmp = line.split('\t')
			for i in range(1, len(tmp)):
				if tmp[i] == '4':
					pos.append(tmp[0])
	return pos

def personal_concern():
	#first we want to get a list of LIWC
	index = 0
	pos = list()

	liwc = open(curr_file_path + 'LIWC2007_English080730.dic').read().split('\n')
	boo = False

	for line in liwc:
		if line == "%":
			boo = not boo
			continue
		if boo:
			continue
		else:
			tmp = line.split('\t')
			for i in range(1, len(tmp)):
				if tmp[i] == '354' or tmp[i] == '356'or tmp[i] == '359' or tmp[i] == '360' or tmp[i] == '357' or tmp[i] == '358':
					pos.append(tmp[0])
	return pos

def informal():
	#first we want to get a list of LIWC
	index = 0
	pos = list()

	liwc = open(curr_file_path + 'LIWC2007_English080730.dic').read().split('\n')
	boo = False

	for line in liwc:
		if line == "%":
			boo = not boo
			continue
		if boo:
			continue
		else:
			tmp = line.split('\t')
			for i in range(1, len(tmp)):
				if tmp[i] == '462' or tmp[i] == '463'or tmp[i] == '464' or tmp[i] == '22':
					pos.append(tmp[0])
	return pos


def certain():
	#first we want to get a list of LIWC
	index = 0
	pos = list()

	liwc = open(curr_file_path + 'LIWC2007_English080730.dic').read().split('\n')
	boo = False

	for line in liwc:
		if line == "%":
			boo = not boo
			continue
		if boo:
			continue
		else:
			tmp = line.split('\t')
			for i in range(1, len(tmp)):
				if tmp[i] == '136':
					pos.append(tmp[0])
	return pos
