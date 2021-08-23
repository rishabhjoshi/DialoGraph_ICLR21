# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser
from cocoa.src.basic.scenario_db import ScenarioDB, Scenario, add_scenario_arguments
from cocoa.src.basic.event import Event
from cocoa.src.basic.util import write_json, read_json
import datetime
import ast

__author__ = 'anushabala'  # modified by 'eahn1'



def add_visualization_arguments(parser):
	parser.add_argument('--html-output', help='Name of directory to write HTML report to')
	parser.add_argument('--viewer-mode', action='store_true', help='Output viewer instead of single html')
	parser.add_argument('--css-file', default='chat_viewer/css/my.css', help='css for tables/scenarios and chat logs')

QUESTIONS = ['n00_gender', 'n01_i_understand', 'n02_cooperative', 'n03_human', 'n04_understand_me', 'n05_chat', 'n06_texts', 'n07_tech', 'n08_learn_spa', 'n09_learn_eng', 'n10_age', 'n11_ability_spa', 'n12_ability_eng', 'n13_country', 'n14_online_spa', 'n15_online_eng', 'n16_online_mix', 'n17_comments']
TEXT_ONLY = ['n08_learn_spa', 'n09_learn_eng', 'n10_age', 'n13_country', 'n17_comments']

# Canonical names to be displayed
AGENT_NAMES = {'human': 'human', 'rule_bot': 'rule-based', 'dynamic-neural': 'DynoNet', 'static-neural': 'StanoNet'}


def get_scenario(chat):
	scenario = Scenario.from_dict(None, chat['scenario'])
	return scenario


def render_chat(chat, agent=None, partner_type='human'):
	events = [Event.from_dict(e) for e in chat["events"]]

	if len(events) == 0:
		return False, None

	chat_html = ['<div class=\"chatLog\">',
				'<div class=\"divTitle\"> Chat Log </div>',
				'<table class=\"chat\">']
	agent_str = {0: '', 1: ''}

	# Used for visualizing chat during debugging
	if agent is not None:
		agent_str[agent] = 'Agent %d (you)' % agent
		agent_str[1 - agent] = 'Agent %d (%s)' % (1 - agent, partner_type)
	elif 'agents' in chat and chat['agents']:
		for agent in (0, 1):
			agent_str[agent] = 'Agent %d (%s)' % (agent, AGENT_NAMES[chat['agents'][str(agent)]])
	else:
		for agent in (0, 1):
			agent_str[agent] = 'Agent %d (%s)' % (agent, 'unknown')

	for event in events:
		t = datetime.datetime.fromtimestamp(float(event.time)).strftime('%Y-%m-%d %H:%M:%S')
		a = agent_str[event.agent]
		if event.action == 'message':
			s = event.data
		elif event.action == 'select':
			# s = 'SELECT (' + ' || '.join(event.data.values()) + ')'
			event_vals = ast.literal_eval(event.data).values()
			# import pdb; pdb.set_trace()
			s = 'SELECT (' + ' || '.join(event_vals) + ')'
		row = '<tr class=\"agent%d\">\
				<td class=\"time\">%s</td>\
				<td class=\"agent\">%s</td>\
				<td class=\"message\">%s</td>\
				</tr>' % (event.agent, t, a, s)
		chat_html.append(row)

	chat_html.extend(['</table>', '</div>'])

	completed = False if chat["outcome"] is None or chat["outcome"]["reward"] == 0 else True
	return completed, chat_html


def _render_response(response, agent_id, agent):
	html = []
	html.append('<table class=\"response%d\">' % agent_id)
	# html.append('<tr><td colspan=\"4\" class=\"agentLabel\">Response to agent %d (%s)</td></tr>' % (agent_id, AGENT_NAMES[agent]))
	html.append('<tr><td colspan=\"2\" class=\"agentLabel\">Response to agent %d (%s)</td></tr>' % (agent_id, AGENT_NAMES[agent]))
	# html.append('<tr>%s</tr>' % (''.join(['<th>%s</th>' % x for x in ('Question', 'Mean', 'Response', 'Justification')])))
	html.append('<tr>%s</tr>' % (''.join(['<th>%s</th>' % x for x in ('Question', 'Response')])))
	for question in QUESTIONS:
		if question not in response:  # or question == 'comments' or question.endswith('text'):
			continue

		answer = response[question]
		# print type(answer)
		if type(answer) == unicode:
			answer = answer.encode('utf-8')
		html.append('<tr><td>{}</td><td>{}</td></tr>'.format(question, answer))

	html.append('</table>')
	return html


def render_scenario(scenario):
	html = ["<div class=\"scenario\">"]
	html.append('<div class=\"divTitle\">Scenario %s</div>' % scenario.uuid)
	for (idx, kb) in enumerate(scenario.kbs):
		kb_dict = kb.to_dict()
		attributes = [attr.name for attr in scenario.attributes]
		scenario_alphas = scenario.alphas
		if len(scenario_alphas) == 0:
			scenario_alphas = ['default' * len(scenario.attributes)]
		alphas = dict((attr.name, alpha) for (attr, alpha) in zip(scenario.attributes, scenario_alphas))
		html.append("<div class=\"kb%d\"><table><tr>"
					"<td colspan=\"%d\" class=\"agentLabel\">Agent %d</td></tr>" % (idx, len(attributes), idx))

		for attr in attributes:
			html.append("<th>%s (%.1f)</th>" % (attr, alphas[attr]))
		html.append("</tr>")

		for item in kb_dict:
			html.append("<tr>")
			for attr in attributes:
				html.append("<td>%s</td>" % item[attr])
			html.append("</tr>")

		html.append("</table></div>")

	html.append("</div>")
	return html


def render_response(responses, agent_dict):
	html_lines = ["<div class=\"survey\">"]
	html_lines.append('<div class=\"divTitle\">Survey</div>')
	for agent_id, response in responses.items():
		html_lines.append('<div class=\"response\">')
		response_html = _render_response(response, int(agent_id), agent_dict[agent_id])
		html_lines.extend(response_html)
		html_lines.append("</div>")
	html_lines.append("</div>")
	return html_lines


def visualize_chat(chat, agent=None, partner_type='Human', responses=None, id_=None):
	completed, chat_html = render_chat(chat, agent, partner_type)
	if chat_html is None:
		return False, None

	# html_lines = []
	html_lines = ['<p>CHAT_ID: {}</p>'.format(chat['uuid']), '<p>SYTLE: {}</p>'.format(chat['scenario']['styles'])]

	scenario_html = render_scenario(get_scenario(chat))
	html_lines.extend(scenario_html)

	html_lines.extend(chat_html)

	if responses:
		dialogue_id = chat['uuid']
		agents = chat['agents']
		# allow for possibility of chat without completing survey
		if dialogue_id in responses:
			response_html = render_response(responses[dialogue_id], agents)
			html_lines.extend(response_html)

	return completed, html_lines


def aggregate_chats(transcripts, responses=None, css_file=None):
	html = ['<!DOCTYPE html>','<html>',
			'<head><style>table{ table-layout: fixed; width: 600px; border-collapse: collapse; } '
			'tr:nth-child(n) { border: solid thin;}</style></head><body>']

	# inline css
	if css_file:
		html.append('<style>')
		with open(css_file, 'r') as fin:
			for line in fin:
				html.append(line.strip())
		html.append('</style>')

	completed_chats = []
	incomplete_chats = []
	total = 0
	num_completed = 0
	for (idx, chat) in enumerate(transcripts):
		completed, chat_html = visualize_chat(chat, responses=responses, id_=idx)
		if completed:
			num_completed += 1
			completed_chats.extend(chat_html)
			completed_chats.append('</div>')
			completed_chats.append("<hr>")
		else:
			if chat_html is not None:
				incomplete_chats.extend(chat_html)
				incomplete_chats.append('</div>')
				incomplete_chats.append("<hr>")
		total += 1

	html.extend(['<h3>Total number of chats: %d</h3>' % total,
				 '<h3>Number of chats completed: %d</h3>' % num_completed,
				 '<hr>'])
	html.extend(completed_chats)
	html.extend(incomplete_chats)
	html.append('</body></html>')
	return html


def visualize_transcripts(html_output, transcripts, responses=None, css_file=None):
	if not os.path.exists(os.path.dirname(html_output)) and len(os.path.dirname(html_output)) > 0:
		os.makedirs(os.path.dirname(html_output))

	html_lines = aggregate_chats(transcripts, responses, css_file)

	outfile = open(html_output, 'w')
	for line in html_lines:
		print line
		if type(line) == str:
			line = line.decode('utf-8')

		line = line.encode('utf-8')
		outfile.write(line + "\n")
	outfile.close()


# @param: chatids = list of chat ids to display
def visualize_chats(chat_file, surv_file, chatids, html_output):
	chats_list = []
	# filter only for chats given
	new_chats_list = []
	survs_dict = {}

	with open(chat_file, 'r') as f:
		for filename in f.readlines():
			chats_list.extend(read_json(filename.strip()))

	for chat_dict in chats_list:
		if chat_dict['uuid'] in chatids:
			new_chats_list.append(chat_dict)

	# filter survey
	with open(surv_file, 'r') as f:
		for filename in f.readlines():
			for k, v in read_json(filename.strip())[1].items():
				if k in chatids:
					survs_dict[k] = v

	css_file = 'cocoa/chat_viewer/css/my.css'
	visualize_transcripts(html_output, new_chats_list, responses=survs_dict, css_file=css_file)
