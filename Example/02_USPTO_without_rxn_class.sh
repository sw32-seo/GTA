#!/bin/bash
dataset=USPTO-50k_no_rxn
model_name=reproduce_wo_rxn_class

python graph_mask_max.py -data data/${dataset}

python preprocess.py -train_src data/${dataset}/src-train.txt \
                     -train_tgt data/${dataset}/tgt-train.txt \
                     -valid_src data/${dataset}/src-val.txt \
                     -valid_tgt data/${dataset}/tgt-val.txt \
                     -save_data data/${dataset}/${dataset} \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab

python  train.py -data data/${dataset}/${dataset} \
                 -save_model experiments/${dataset}_${model_name} \
                 -seed 2020 -gpu_ranks 0 \
                 -save_checkpoint_steps 1000  -keep_checkpoint 11 \
                 -train_steps 400000 -valid_steps 1000 -report_every 1000 \
                 -param_init 0 -param_init_glorot \
                 -batch_size 4096 -batch_type tokens -normalization tokens \
                 -dropout 0.3 -max_grad_norm 0 -accum_count 4 \
                 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 \
                 -decay_method noam -warmup_steps 8000  \
                 -learning_rate 2 -label_smoothing 0.0 \
                 -enc_layers 6 -dec_layers 6 -rnn_size 256 -word_vec_size 256 \
                 -encoder_type transformer -decoder_type transformer \
                 -share_embeddings -position_encoding -max_generator_batches 0 \
                 -global_attention general -global_attention_function softmax \
                 -self_attn_type scaled-dot -max_relative_positions 4 \
                 -heads 8 -transformer_ff 2048 -max_distance 1 2 3 4 \
                 -early_stopping 40 -alpha 1.0 \
                 -tensorboard -tensorboard_log_dir runs/${dataset}_${model_name} 2>&1 | tee train_$model_name.log

python Generate_test_prediction.py data/${dataset}/src-test.txt -model_path experiments/${dataset}_${model_name}

python parse/parse_output.py -input_file experiments/${dataset}_${model_name}/pred/output \
                             -target_file data/${dataset}/tgt-test.txt -beam_size 10
