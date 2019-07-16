#!/usr/bin/env bash
python generate.py -snapshot graph-nvp-final.npz \
--gpu -1 \
--data_name qm9 \
--data_dir data \
--hyperparams-path graphnvp-params.json \
--batch-size 1000 \
--model_dir models/qm9 \
--temperature 0.85 \
--n_experiments 5 \
--delta 0.05
