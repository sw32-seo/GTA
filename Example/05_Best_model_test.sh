#!/bin/bash
dataset=USPTO-50k_no_rxn
model_name=Best_model

python graph_mask_max.py -data data/${dataset} -test

python Generate_test_prediction.py -src data/${dataset}/src-test.txt -model_path experiments/${dataset}_${model_name}

python parse_output.py -input_file experiments/${dataset}_${model_name}/pred/output \
                             -target_file data/${dataset}/tgt-test.txt -beam_size 50