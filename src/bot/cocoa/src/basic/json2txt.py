import json
import sys

def write_file_full(infile, outfile):
	read_list = json.load(open(infile))
	out_list = []

	for chat in read_list:
		idx = ['B > ','H\t']
		if chat['agents']['0']=='human':
			idx = ['H\t','B > ']
		for event in chat['events']:
			if not event['action']=='message': continue # not 'select'
			msg = event['data'].encode('utf-8')
			agent = event['agent']
			out_list.append("{}{}".format(idx[agent], msg))
		out_list.append("-"*10)

	writer = open(outfile, 'a+')
	for line in out_list:
		writer.write(line + "\n")
	writer.close()

def write_file_bot_only(infile, outfile):
	read_list = json.load(open(infile))
	out_list = []

	for chat in read_list:
		idx = 0
		if chat['agents']['0']=='human':
			idx = 1
		for event in chat['events']:
			if not event['action']=='message': continue # not 'select'
			msg = event['data']
			agent = event['agent']
			if agent==idx:
				out_list.append(msg)

	writer = open(outfile, 'w')
	for line in out_list:
		writer.write(line + "\n")
	writer.close()

dyno_file = "data/chat_prev/human-dynamic-neural_transcripts.json"
rule_file = "data/chat_prev/human-rulebased_transcripts.json"

# write_file_full(dyno_file, "data/chat_prev/hum_dyno.txt")
# write_file_full(rule_file, "data/chat_prev/hum_rule.txt")
# write_file_bot_only(dyno_file, "data/chat_prev/bot_dyno.txt")
# write_file_bot_only(rule_file, "data/chat_prev/bot_rule.txt")

infile = sys.argv[1]
outfile = sys.argv[2]
write_file_full(infile, outfile)