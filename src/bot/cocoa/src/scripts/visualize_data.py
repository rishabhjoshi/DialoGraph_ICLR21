# -*- coding: utf-8 -*-
from src.basic.util import read_json
import pdb
import ast

__author__ = 'anushabala'

import os
from argparse import ArgumentParser
from src.basic.scenario_db import ScenarioDB, Scenario, add_scenario_arguments
from src.basic.schema import Schema
from src.basic.event import Event
from src.basic.util import write_json, read_json
import numpy as np
import json
import datetime
from collections import defaultdict

def add_visualization_arguments(parser):
    parser.add_argument('--html-output', help='Name of directory to write HTML report to')
    parser.add_argument('--viewer-mode', action='store_true', help='Output viewer instead of single html')
    parser.add_argument('--css-file', default='chat_viewer/css/my.css', help='css for tables/scenarios and chat logs')
    # parser.add_argument('--css-file', default='src/web/static/css/bootstrap4.min.css', help='css for tables/scenarios and chat logs')

#questions = ['fluent', 'fluent_text', 'correct', 'correct_text', 'cooperative', 'cooperative_text', 'strategic', 'strategic_text', 'humanlike', 'humanlike_text', 'comments']
# QUESTIONS = ['fluent', 'correct', 'cooperative', 'humanlike']
QUESTIONS = ['n00_gender', 'n01_i_understand', 'n02_cooperative', 'n03_human', 'n04_understand_me', 'n05_chat', 'n06_texts', 'n07_tech', 'n08_learn_spa', 'n09_learn_eng', 'n10_age', 'n11_ability_spa', 'n12_ability_eng', 'n13_country', 'n14_online_spa', 'n15_online_eng', 'n16_online_mix', 'n17_comments']
MYQUESTIONS = ['n00_gender', 'n01_i_understand', 'n02_cooperative', 'n03_human', 'n04_understand_me', 'n05_chat', 'n10_age', 'n11_ability_spa', 'n13_country', 'n17_comments']
TEXT_ONLY = ['n08_learn_spa', 'n09_learn_eng', 'n10_age', 'n13_country', 'n17_comments']

# Canonical names to be displayed
# AGENT_NAMES = {'human': 'human', 'rulebased': 'rule-based', 'dynamic-neural': 'DynoNet', 'static-neural': 'StanoNet'}
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
        if type(answer) == str:
            answer = answer#.encode('utf-8')
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

    html_lines = []

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

def get_data(chat, responses, idx):
    events = [Event.from_dict(e) for e in chat["events"]]
    if len(events) == 0:
        return False, None
    agent_str = {0: '', 1: ''}
    # Used for visualizing chat during debugging
    for agent in (0,1):
        agent_str[agent] = AGENT_NAMES[chat['agents'][str(agent)]]

    tot_turns = len(events)
    num_words_per_turn = []
    model = ''
    dialogue_id = chat['uuid']
    first_turn = 1 
    utterances = []
    for event in events:
        a = agent_str[event.agent]
        if event.action == 'message':
            s = event.data
            if first_turn:
                if s == 'Hello.': model = 'hed'
                elif s == 'Hello..': model = 'fst'
                elif s == 'Hello!': model = 'rnn'
                elif s == 'Hello!!': model = 'transformer'
                elif s == 'Hello!!!': model = 'graph'
                else:
                    pdb.set_trace()
                first_turn = 0

            if a == 'rule-based':
                num_words_per_turn.append(len(s.split()))
        else:
            s = event.action
        utterances.append(a + ':' + s.replace(',','').replace('\n',''))
    utterances = ' ### '.join(utterances)
    num_words_per_turn = np.mean(num_words_per_turn)
    from collections import defaultdict as ddict
    sur = ddict(str)
    completed = 1
    res = [model, dialogue_id, 0, tot_turns, num_words_per_turn, utterances]
    if dialogue_id in responses:
        response = responses[dialogue_id]
        for agent_id, r in response.items():
            for question in QUESTIONS:
                if question not in r:  # or question == 'comments' or question.endswith('text'):
                    continue
                answer = r[question]
                if type(answer) == str:
                    answer = answer.replace(',','').replace('\n','')
                sur[question] = answer
        for question in MYQUESTIONS:
            res.append(sur[question])
    else:
        completed = 0
    res = [completed] + res

    return res


def get_data_stats(transcripts, survey_responses):
    '''
    dumps in format
    model,scenarioID,0,gender,IunderstandTask,persuasive,human,understandme,coherent,age,specialQuesAns4,country,comments,num_of_turns,avg_num_of_turns,completed
    '''
    completed_chats, incomplete_chats = [], []
    for (idx, chat) in enumerate(transcripts):
        res = get_data(chat, survey_responses, idx)
        print (','.join([str(r) for r in res]))

    return None


def visualize_transcripts(html_output, transcripts, responses=None, css_file=None):
    if not os.path.exists(os.path.dirname(html_output)) and len(os.path.dirname(html_output)) > 0:
        os.makedirs(os.path.dirname(html_output))

    html_lines = aggregate_chats(transcripts, responses, css_file)
    data_stats_which_you_can_copy_in_excel = get_data_stats(transcripts, responses)

    outfile = open(html_output, 'w')
    for line in html_lines:
        #print (line)
        if type(line) == str:
            line = line#.decode('utf-8')

        line = line#.encode('utf-8')
        outfile.write(line + "\n")
    outfile.close()


def write_chat_htmls(transcripts, outdir, responses=None):
    outdir = os.path.join(outdir, 'chat_htmls')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for chat in transcripts:
        dialogue_id = chat['uuid']
        _, chat_html = visualize_chat(chat, responses=responses)
        if not chat_html:
            continue
        with open(os.path.join(outdir, dialogue_id+'.html'), 'w') as fout:
            # For debugging: write complete html file
            #fout.write("<!DOCTYPE html>\
            #        <html>\
            #        <head>\
            #        <link rel=\"stylesheet\" type\"text/css\" href=\"../css/my.css\"\
            #        </head>")
            fout.write('\n'.join(chat_html))#.encode('utf-8'))
            #fout.write("</html>")

def write_metadata(transcripts, outdir, responses=None):
    metadata = {'data': []}
    for chat in transcripts:
        if len(chat['events']) == 0:
            continue
        row = {}
        row['dialogue_id'] = chat['uuid']
        row['scenario_id'] = chat['scenario_uuid']
        scenario = get_scenario(chat)
        row['num_items'] = len(scenario.kbs[0].items)
        row['num_attrs'] = len(scenario.attributes)
        row['outcome'] = 'fail' if chat['outcome']['reward'] == 0 else 'success'
        row['agent0'] = AGENT_NAMES[chat['agents']['0']]
        row['agent1'] = AGENT_NAMES[chat['agents']['1']]
        if responses:
            dialogue_response = responses[chat['uuid']]
            question_scores = defaultdict(list)
            for agent_id, scores in dialogue_response.items():
                for question in QUESTIONS:
                    question_scores[question].extend(scores[question])
            for question, scores in question_scores.items():
                row[question] = np.mean(scores)
        metadata['data'].append(row)
    write_json(metadata, os.path.join(outdir, 'metadata.json'))

def write_viewer_data(html_output, transcripts, responses=None):
    if not os.path.exists(html_output):
        os.makedirs(html_output)
    write_metadata(transcripts, html_output, responses)
    write_chat_htmls(transcripts, html_output, responses)


if __name__ == "__main__":
    parser = ArgumentParser()
    add_scenario_arguments(parser)
    add_visualization_arguments(parser)
    parser.add_argument('--transcripts', type=str, default='transcripts.json', help='Path to json file containing chats')
    parser.add_argument('--survey_file', type=str, default=None, help='Path to json file containing survey')

    args = parser.parse_args()
    schema = Schema(args.schema_path)
    # scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    transcripts = read_json(args.transcripts)
    # import pdb; pdb.set_trace()
    survey = read_json(args.survey_file)[1]
    html_output = args.html_output

    if args.viewer_mode:
        # External js and css
        write_viewer_data(html_output, transcripts, responses=survey)
    else:
        # Inline style
        visualize_transcripts(html_output, transcripts, responses=survey, css_file=args.css_file)
