# out_name=soc1scratch
# out_folder=turk/amt_soc1_0724

# chat_db=${out_folder}/0720_social.db
# scenarios=data/scenarios_0720_social.json
# schema=data/schema_0720_social.json

# batch_file=${out_folder}/0720_social_batch.csv
# batch_arg="--batch-results $batch_file"

# out_name=scratch-cont2
# out_folder=turk/amt_cont2_0716

# chat_db=${out_folder}/cont_0716.db
# scenarios=data/scenarios_0614_cont.json
# schema=data/schema_0614_cont.json

# batch_file=${out_folder}/cont_0718_batch.csv
# batch_arg="--batch-results $batch_file"


PYTHONPATH=. python src/web/dump_db_neg.py --db $chat_db --output ${out_folder}/${out_name}_chat.json  --schema-path $schema --scenarios-path $scenarios --surveys ${out_folder}/${out_name}_surv.json $batch_arg