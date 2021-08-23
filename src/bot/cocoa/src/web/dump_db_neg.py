import sqlite3
import json
import math
import os
from argparse import ArgumentParser

# from cocoa.core.schema import Schema
from src.basic.schema import Schema
# from cocoa.core.scenario_db import add_scenario_arguments, ScenarioDB
from src.basic.scenario_db import add_scenario_arguments, ScenarioDB
# from cocoa.core.util import read_json, write_json
from src.basic.util import read_json, write_json

# Task-specific modules
# from web.main.db_reader import DatabaseReader
from src.web.db_reader_neg import DatabaseReader
# from core.scenario import Scenario


def read_results_csv(csv_file):
    '''
    Return a dict from mturk_code to worker_id.
    '''
    import csv
    reader = csv.reader(open(csv_file, 'r'))
    header = reader.next()
    worker_idx = header.index('WorkerId')
    code_idx = header.index('Answer.surveycode')
    d = {}
    for row in reader:
        workerid = row[worker_idx]
        code = row[code_idx]
        d[code] = workerid
    return d


def read_results_json(jsonfile):
    '''
    Return a dict from mturk_code to worker_id. For Figure8 JSON files.
    '''
    d = {}

    json_dict = json.load(open(jsonfile))
    for worker in json_dict['results']['judgments']:
        code = None
        for ques, answer in worker['data'].items():
            answer = answer.strip()
            if answer.startswith('MTURK'):
                code = answer

        if code is None:
            continue

        d[code] = worker['worker_id']

    return d


# update: code_to_wid param became list to account for several batch files
def chat_to_worker_id(cursor, code_to_wid_list):
    '''
    chat_id: {'0': worker_id, '1': worker_id}
    worker_id is None means it's a bot
    '''
    d = {}
    cursor.execute('SELECT chat_id, agent_ids FROM chat')
    for chat_id, agent_uids in cursor.fetchall():
        agent_wid = {}
        agent_uids = eval(agent_uids)
        for agent_id, agent_uid in agent_uids.items():
            if not (isinstance(agent_uid, str)):  # and agent_uid.startswith('U_')):
                agent_wid[agent_id] = None
            else:
                cursor.execute('''SELECT mturk_code FROM mturk_task WHERE name=?''', (agent_uid,))
                res = cursor.fetchall()
                if len(res) > 0:
                    mturk_code = res[0][0]
                    for code_to_wid in code_to_wid_list:
                        if mturk_code not in code_to_wid:
                            continue
                        else:
                            agent_wid[agent_id] = code_to_wid[mturk_code]
                            agent_wid["mturk_code"] = mturk_code
        d[chat_id] = agent_wid
    return d


# @eahn1: modified to have unique outfile json files (prevent rewrite existing)
# batch_results has type=list
def log_worker_id_to_json(db_path, batch_results):
    '''
    {chat_id: {'0': worker_id; '1': worker_id}}
    '''
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    code_to_wid_list = []
    for batch_result in batch_results:
        if batch_result.endswith('csv'):
            code_to_wid = read_results_csv(batch_result)
        else:  # fig8 JSON
            code_to_wid = read_results_json(batch_result)

        code_to_wid_list.append(code_to_wid)

    worker_ids = chat_to_worker_id(cursor, code_to_wid_list)

    output_dir = os.path.dirname(batch_results[0])
    # outfile_name = os.path.splitext(os.path.basename(batch_results[0]))[0] + '_worker_ids.json'
    outfile_name = 'worker_ids.json'
    outfile_path = os.path.join(output_dir, outfile_name)
    write_json(worker_ids, outfile_path)


# @eahn1: modified method from original file: dump_events_to_json.py
def log_surveys_to_json(cursor, surveys_file):
    questions = ['n00_gender', 'n01_i_understand', 'n02_cooperative', 'n03_human', 'n04_understand_me', 'n05_chat', 'n06_texts', 'n07_tech', 'n08_learn_spa', 'n09_learn_eng', 'n10_age', 'n11_ability_spa', 'n12_ability_eng', 'n13_country', 'n14_online_spa', 'n15_online_eng', 'n16_online_mix', 'n17_comments']
    # conn = sqlite3.connect(db_path)
    # cursor = conn.cursor()
    cursor.execute('''SELECT * FROM survey''')
    logged_surveys = cursor.fetchall()
    survey_data = {}
    agent_types = {}

    for survey in logged_surveys:
        # print survey
        (userid, cid, _, n00_gender, n01_i_understand, n02_cooperative, n03_human, n04_understand_me, n05_chat, n06_texts, n07_tech, n08_learn_spa, n09_learn_eng, n10_age, n11_ability_spa, n12_ability_eng, n13_country, n14_online_spa, n15_online_eng, n16_online_mix, n17_comments) = survey
        responses = dict(zip(questions, [n00_gender, n01_i_understand, n02_cooperative, n03_human, n04_understand_me, n05_chat, n06_texts, n07_tech, n08_learn_spa, n09_learn_eng, n10_age, n11_ability_spa, n12_ability_eng, n13_country, n14_online_spa, n15_online_eng, n16_online_mix, n17_comments]))
        cursor.execute('''SELECT agent_types, agent_ids FROM chat WHERE chat_id=?''', (cid,))
        chat_result = cursor.fetchone()
        agents = json.loads(chat_result[0])
        agent_ids = json.loads(chat_result[1])
        agent_types[cid] = agents
        if cid not in survey_data.keys():
            survey_data[cid] = {0: {}, 1: {}}
        partner_idx = 0 if agent_ids['1'] == userid else 1
        survey_data[cid][partner_idx] = responses

    json.dump([agent_types, survey_data], open(surveys_file, 'w'))


if __name__ == "__main__":
    parser = ArgumentParser()
    add_scenario_arguments(parser)
    parser.add_argument('--db', type=str, required=True, help='Path to database file containing logged events')
    parser.add_argument('--output', type=str, required=True, help='File to write JSON examples to.')
    parser.add_argument('--uid', type=str, nargs='*', help='Only print chats from these uids')
    parser.add_argument('--surveys', type=str, help='If provided, writes a file containing results from user surveys.')
    parser.add_argument('--batch-results', type=str, nargs='*', help='If provided, write a mapping from chat_id to worker_id for several files')
    args = parser.parse_args()

    schema = Schema(args.schema_path)
    # scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    DatabaseReader.dump_chats(cursor, scenario_db, args.output, args.uid)
    if args.surveys:
        # DatabaseReader.dump_surveys(cursor, args.surveys)
        log_surveys_to_json(cursor, args.surveys)
    # TODO: move this to db_reader
    if args.batch_results:
        log_worker_id_to_json(args.db, args.batch_results)
