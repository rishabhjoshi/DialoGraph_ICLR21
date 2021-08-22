import os
curr_file_path = os.path.dirname(os.path.abspath(__file__)) + '/'
def get_dominance_valence_arousal():
	#read csv
	wb = open(curr_file_path + 'BRM-emot-submit.csv').read().split("\n")
	header = wb[0].split(",")
	dominance = dict()
	valence = dict()
	arousal = dict()
	for i in range(1, len(wb)):
		tmp = wb[i].split(",")
		dominance[tmp[1]] = float(tmp[8])
		valence[tmp[1]] = float(tmp[2])
		arousal[tmp[1]] = float(tmp[5])
	return dominance, valence, arousal


