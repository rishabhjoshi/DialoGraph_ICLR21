import json
import csv
from collections import defaultdict
from argparse import ArgumentParser


""" Script that for 1 batch, reads pieces of everything, outputs TSV file.
	TSV to be copied to spreadsheet and processed manually later.

	Example run from cocoa/:
	python turk/process_survey.py --workers turk/fig8_0_all4/worker_ids.json --chats turk/fig8_0_all4/fig8_0_chat.json --surveys turk/fig8_0_all4/fig8_0_surv.json --crowdfiles turk/fig8_0_all4/job_0725.json turk/fig8_0_all4/job_0727.json --outfile turk/fig8_0_all4/fig8_0_qual.tsv

	Modified: 29 July 2018
"""


def time_to_num(time_string):
	''' ex. Mon Jun 18 02:41:52 PDT 2018
		transforms to
		06-18 02:41:52
	'''
	new_str = time_string[4:-9]  # Jun 18 02:41:52
	month = new_str[:3]
	month_str = '00'
	# import pdb; pdb.set_trace()
	if month == 'Jun':
		month_str = '06'
	elif month == 'Jul':
		month_str = '07'
	elif month == 'Aug':
		month_str = '08'
	elif month == 'Sep':
		month_str = '09'

	return '{}-{}'.format(month_str, new_str[4:])


# grab duration time and time submitted for each chat_id in mturk CSV file
def read_mturk_csv(csvfile):
	reader = csv.reader(open(csvfile, 'r'))
	header = reader.next()
	# worker_idx = header.index('WorkerId')
	code_idx = header.index('Answer.surveycode')
	duration_idx = header.index('WorkTimeInSeconds')
	submit_idx = header.index('SubmitTime')
	mturk_metadata = defaultdict(dict)

	for row in reader:
		# workerid = row[worker_idx]
		code = row[code_idx]
		mturk_metadata[code]["duration"] = float(row[duration_idx]) / 60
		mturk_metadata[code]["submit"] = time_to_num(row[submit_idx])

	return mturk_metadata


# grab duration time and time submitted for each chat_id in fig8 JSON file
def read_fig8_json(jsonfile):
	mturk_metadata = defaultdict(dict)

	json_dict = json.load(open(jsonfile))
	for worker in json_dict['results']['judgments']:
		code = None
		for ques, answer in worker['data'].items():
			answer = answer.strip()
			if answer.startswith('MTURK'):
				code = answer

		if code is None:
			continue

		submitted = worker['created_at'].split('+')[0]  # 2018-07-25T20:35:08+00:00
		submitted = submitted.replace('2018-', '').replace('T', ' ')  # 07-25 20:35:08
		started = worker['started_at'].split('+')[0]
		duration = (int(submitted.split(':')[1]) + 60 - int(started.split(':')[1])) % 60
		mturk_metadata[code]["duration"] = duration  # minutes (type=int)
		mturk_metadata[code]["submit"] = submitted  # (type=str)

	return mturk_metadata


def main():
	parser = ArgumentParser('process survey and chats from cocoa')
	parser.add_argument(
		'--workers', type=str, required=True,
		help='File path to worker ID json')
	parser.add_argument(
		'--chats', type=str, required=True,
		help='File path to chats json')
	parser.add_argument(
		'--surveys', type=str, required=True,
		help='File path to surveys json')
	parser.add_argument(
		'--crowdfiles', type=str, nargs='+', required=True,
		help='File path to either mturk CSV or fig8 JSON')
	parser.add_argument(
		'--outfile', type=str, required=True,
		help='Outfile path. TSV of full info')
	args = parser.parse_args()

	worker_ids_file = args.workers
	chat_file = args.chats
	survey_file = args.surveys
	crowd_files = args.crowdfiles
	outfile =  args.outfile # "19struct_qual.tsv"

	for i, crowd_file in enumerate(crowd_files):
		if crowd_file.endswith('csv'):  # mturk
			this_mturk_metadata = read_mturk_csv(crowd_file)
		else:  # fig8
			this_mturk_metadata = read_fig8_json(crowd_file)

		if i == 0:
			mturk_metadata = this_mturk_metadata
		else:  # beyond 1st crowd_file
			mturk_metadata.update(this_mturk_metadata)

	worker_ids = json.load(open(worker_ids_file))

	chat_info = json.load(open(chat_file))
	survey_info = json.load(open(survey_file))
	# import pdb; pdb.set_trace()
	new_dict = defaultdict(dict)
	worker_to_chat_dict = defaultdict(list)

	for chat_id, items in worker_ids.items():
		if not any([full_chat['uuid'] == chat_id for full_chat in chat_info]):
			continue

		if len(items) == 1:
			worker_id = "[none]"
			new_dict[chat_id] = {}
			# continue

		else:
			worker_id = items["1"]
			if items["1"] is None:
				worker_id = items["0"]

			hit_code = items["mturk_code"]
			new_dict[chat_id] = {}
			new_dict[chat_id]["duration"] = mturk_metadata[hit_code]["duration"]
			new_dict[chat_id]["submit"] = mturk_metadata[hit_code]["submit"]

		worker_to_chat_dict[worker_id].append(chat_id)

	# for worker_id, chatidlist in worker_to_chat_dict.items():
	# 	if len(chatidlist) > 1:
	# 		print worker_id, chatidlist

	# go through list of chat dicts
	for full_chat in chat_info:
		this_chatid = full_chat["uuid"]
		this_style = full_chat["scenario"]["styles"]
		outcome = full_chat["outcome"]["reward"]  # 0 or 1

		new_dict[this_chatid]["style"] = this_style
		new_dict[this_chatid]["outcome"] = outcome

	# go through survey
	for chat_id, items in survey_info[1].items():
		values_dict = items["0"]
		if len(items["0"]) == 0:
			values_dict = items["1"]

		new_dict[chat_id]["questions"] = values_dict

	# import pdb; pdb.set_trace()
	# print new_dict[u'C_326dc1a4bb484d469981966acd08d6f0'].keys()
	# print new_dict[u'C_326dc1a4bb484d469981966acd08d6f0']['questions'].keys()

	header = ['chat_id', 'worker_id']
	cats = ['style', 'submit', 'duration', 'outcome']
	questions = [u'n00_gender', u'n01_i_understand', u'n02_cooperative', u'n03_human', u'n04_understand_me', u'n05_chat', u'n06_texts', u'n07_tech', u'n08_learn_spa', u'n09_learn_eng', u'n10_age', u'n11_ability_spa', u'n12_ability_eng', u'n13_country', u'n14_online_spa', u'n15_online_eng', u'n16_online_mix', u'n17_comments']
	all_cats = header
	all_cats.extend(cats)
	all_cats.extend(questions)

	with open(outfile, 'w') as f:
		f.write('\t'.join(all_cats) + '\n')
		for worker_id, chat_list in worker_to_chat_dict.items():
			for chat_id in chat_list:
				if chat_id == '': continue
				if new_dict[chat_id] == {}: continue
				line = '{}\t{}\t'.format(chat_id, worker_id)
				for cat in cats:
					if cat in new_dict[chat_id]:
						item = new_dict[chat_id][cat]
					else:
						item = ''

					line += '{}\t'.format(item)

				if "questions" not in new_dict[chat_id]:
					continue
					# import pdb; pdb.set_trace()

				for question in questions:
					answer = new_dict[chat_id]["questions"][question]
					# if chat_id == 'C_318bb8836b054d348f41e16435570f90' and question == 'n17_comments':
					# 	import pdb; pdb.set_trace()
					if type(answer) == unicode:
						answer = answer.encode('utf-8')
						answer = answer.replace('\n', '')

					if type(answer) == str:
						answer = answer.replace('\n', '')

					line += '{}\t'.format(answer)

				f.write(line.strip() + '\n')

if __name__ == '__main__':
	main()







