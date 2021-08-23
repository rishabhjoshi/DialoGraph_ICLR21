# random useful things
export PYTHONPATH=.
pip install -r requirements.txt
import ipdb; ipdb.set_trace()

/Users/eahn/.pyenv/versions/2.7.10/Python.framework/Versions/2.7/lib/python2.7/site-packages/fuzzywuzzy/fuzz.py:35: UserWarning: Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning

### command to dump logs to outfiles --surveys and --output
python src/web/dump_db_neg.py --db web_output/2018-01-16/chat_state.db --output web_output/log1_01-16_chats.json --surveys web_output/log1_01-16_surv.json --schema-path data/schema.json --scenarios-path data/scenarios.json

### command to run app
PYTHONPATH=. python src/web/start_app.py --port 5000 --schema-path data/friends-schema.json --scenarios-path data/scenarios.json --config data/web/app_params.json

### mt stuff
Downloaded the following with brew:
* subversion
* libtool
* boost
* xmlrpc-c
* boost-build

### train DynoNet
PYTHONPATH=. python src/main.py --schema-path data/schema.json --scenarios-path data/scenarios.json
--train-examples-paths data/train.json --test-examples-paths data/dev.json --stop-words data/common_words.txt
--min-epochs 10 --checkpoint checkpoint --rnn-type lstm --learning-rate 0.5 --optimizer adagrad
--print-every 50 --model attn-copy-encdec --gpu 1 --rnn-size 100 --grad-clip 0 --num-items 12
--batch-size 32 --stats-file stats.json --entity-encoding-form type --entity-decoding-form type
--node-embed-in-rnn-inputs --msg-aggregation max --word-embed-size 100 --node-embed-size 50
--entity-hist-len -1 --learned-utterance-decay

### change in app_params to HTML
adding to routes/html. Example add 'title_task_span'
* data/web/app_params.json : "task_title_span": "¿Quién es nuestro amigo en común?"
* src/web/start_app.py :`app.config['task_title_span'] = params['task_title_span']`
* src/web/main/routes.py : `title_span=app.config['task_title_span']` with possible unicode handling
* src/web/templates/waiting.html : `<h2>{{title_span}}</h2>`
* any other html template