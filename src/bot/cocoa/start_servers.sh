#!/bin/bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python src/web/start_app.py --schema-path data/friends-schema.json --scenarios-path data/scenarios.json --config data/web/app_params.json --strat_model graph --port 8876
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python src/web/start_app.py --schema-path data/friends-schema.json --scenarios-path data/scenarios.json --config data/web/app_params.json --strat_model hed --port 8877
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python src/web/start_app.py --schema-path data/friends-schema.json --scenarios-path data/scenarios.json --config data/web/app_params.json --strat_model rnn --port 8878
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python src/web/start_app.py --schema-path data/friends-schema.json --scenarios-path data/scenarios.json --config data/web/app_params.json --strat_model fst --port 8879
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python src/web/start_app.py --schema-path data/friends-schema.json --scenarios-path data/scenarios.json --config data/web/app_params.json --strat_model transformer --port 8880
