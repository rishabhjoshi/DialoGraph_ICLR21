#!/bin/bash
#amt_logs=("2020-06-30-19-48-06" "2020-06-30-20-23-18" "2020-07-01-06-53-45" "2020-07-01-07-00-00" "2020-07-01-04-51-35" "2020-07-01-06-48-11" "2020-06-30-20-25-57" "2020-06-30-19-51-40" "2020-07-01-05-58-10" "2020-07-01-04-31-42")
# HED OF AMT LATER
amt_logs=("2020-07-01-18-56-34")

for log_dump in "${amt_logs[@]}"
do
	schema="data/friends-schema.json"
	scenario="data/scenarios.json"

	PYTHONPATH=. python src/web/dump_db_neg.py --db web_output/$log_dump/chat_state.db --output web_output/$log_dump/transcripts.json --schema-path $schema --scenarios-path $scenario --surveys web_output/$log_dump/survey.json 
	PYTHONPATH=. python src/scripts/visualize_data.py --scenarios-path $scenario --schema-path $schema --transcripts web_output/$log_dump/transcripts.json --html-output web_output/$log_dump/chat.html --survey_file web_output/$log_dump/survey.json
	#echo "Dumped web_output/$log_dump"
done
