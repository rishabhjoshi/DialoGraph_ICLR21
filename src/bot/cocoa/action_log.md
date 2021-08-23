### 13 Feb
using [fast_align](https://github.com/clab/fast_align) tool


```sh
py style/format_to_align.py data/chat_prev/en_rule.txt data/chat_prev/sp_rule.txt style/en-sp_rule.txt
cd style/
fast_align -i en-sp_rule.txt -d -o -v > en2sp.align
fast_align -i en-sp_rule.txt -d -o -v -r > sp2en.align
```

### 19 Feb
```sh
fast_align -i en-sp_rule.txt -d -v -o -p probs.en2sp > align.en2sp
fast_align -i en-sp_rule.txt -d -v -o -r -p probs.sp2en > align.sp2en
atools -i align.en2sp -j align.sp2en -c grow-diag-final-and > aligned.gdfa
```

### 20 Feb
* new databases in `data/{names,loc,hobbies,time}.txt`
* new schema: `data/mini_schema.json`
* new scenario: `data/mini_scenarios.json`
```sh
PYTHONPATH=. python src/scripts/generate_schema.py --schema-path data/mini_schema.json
PYTHONPATH=. python src/scripts/generate_scenarios.py --schema-path data/mini_schema.json --scenarios-path data/mini_scenarios.json --num-scenarios 10 --random-attributes --random-items --alphas 0.3 1 3
# to run full with new schema
PYTHONPATH=. python src/web/start_app.py --port 5000 --schema-path data/mini_schema.json --scenarios-path data/mini_scenarios.json --config data/web/app_params.json
```

### 3 April
```sh
PYTHONPATH=. python src/scripts/generate_schema.py --schema-path data/mini_schema.json
PYTHONPATH=. python src/scripts/generate_scenarios.py --schema-path data/mini_schema.json --scenarios-path data/mini_scenarios.json --num-scenarios 10 --random-attributes --random-items --alphas 0.3 1 3 --min-items 8 --max-items 8
```

### 10 April
created:
* data/mini_schema_style.json
* data/mini_scenarios_style.json
added param of style to scenarios. Num of scenarios in json = (--num-scenarios * num_styles) -- styles determined by list in `src/scripts/generate_schema.py`

### 20 May
execute `./mturk_process.sh`

### 20 July
add social
```sh
PYTHONPATH=. python src/scripts/generate_scenarios.py --schema-path data/schema_0720_social.json --scenarios-path data/scenarios_0720_social_20.json --num-scenarios 10 --random-attributes --random-items --alphas 0.3 1 3 --min-items 10 --max-items 10 --num-styles 4
```

### 27 July
make social its own explicit style from generate_schema and simple_session
```sh
python src/scripts/generate_schema.py --schema-path data/schema_0727_all8.json

PYTHONPATH=. python src/scripts/generate_scenarios.py --schema-path data/schema_0727_all8.json --scenarios-path data/scenarios_0727_all8.json --num-scenarios 10 --random-attributes --random-items --alphas 0.3 1 3 --min-items 10 --max-items 10 --num-styles
```

### 29 July
change dump_db_neg.py to process worker ids from SEVERAL files, inclduing type JSON from figure8
```sh
PYTHONPATH=. python src/web/dump_db_neg.py --db turk/fig8_0_all4/fig8_0_all4.db --output turk/fig8_0_all4/fig8_0_chat.json  --schema-path data/schema_0720_social.json --scenarios-path data/scenarios_0720_social.json --surveys turk/fig8_0_all4/fig8_0_surv.json --batch-results turk/fig8_0_all4/job_0725.json turk/fig8_0_all4/job_0727.json
```

update process_survey to handle SEVERAL files, including type JSON. Handles duplicate mturk entries in DB and discards bad chats.
```sh
python turk/process_survey.py --workers turk/fig8_0_all4/worker_ids.json --chats turk/fig8_0_all4/fig8_0_chat.json --surveys turk/fig8_0_all4/fig8_0_surv.json --crowdfiles turk/fig8_0_all4/job_0725.json turk/fig8_0_all4/job_0727.json --outfile turk/fig8_0_all4/fig8_0_qual.tsv
```

### 9 Aug
make random style

### 24 Sept
make mono style {en_mono, sp_mono}
```sh
PYTHONPATH=. python src/scripts/generate_schema.py --schema-path data/schema_0924_mono.json
PYTHONPATH=. python src/scripts/generate_scenarios.py --schema-path data/schema_0924_mono.json --scenarios-path data/scenarios_0924_mono.json --num-scenarios 10 --random-attributes --random-items --alphas 0.3 1 3 --min-items 10 --max-items 10 --num-styles 2
```
