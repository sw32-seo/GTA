# GTA: Graph Truncated Attention for Retrosynthesis

This code is the official implementation of GTA: Graph Truncated Attention for Retrosynthesis paper for AAAI2021


Data can be found: https://drive.google.com/drive/folders/1Q2pZgfUwIricTL6c_I66HHW5ULOnLB1p?usp=sharing

This project is built on top of OpenNMT: https://github.com/OpenNMT/OpenNMT-py

and the work of Chen et al. 2019: https://github.com/iclr-2020-retro/retro_smiles_transformer

To install requirements:

```bash
pip install -r requirements.txt
```

and install rdkit==2019.09.3 by following official page 

https://www.rdkit.org/


To model Best model performance,

```bash
mkdir experiments
```

then please download below folder into experiments/:

https://drive.google.com/drive/folders/1Q2pZgfUwIricTL6c_I66HHW5ULOnLB1p?usp=sharing

and place model and data as 

```bash
experiemnts/USPTO-50k_no_rxn_Best_model/models
data/USPTO-50k_no_rxn
```

and run

```bash
bash Example/05_Best_model_test.sh
```

To generate adjacency matrix and atom mapping (This will take two hours for USPTO-50k_no_rxn dataset):

```bash
dataset=data_name
python graph_mask_max.py -data data/${dataset}
```

To preprocess the data:

```bash
dataset=data_name
python preprocess.py -train_src data/${dataset}/src-train.txt \
                     -train_tgt data/${dataset}/tgt-train.txt \
                     -valid_src data/${dataset}/src-val.txt \
                     -valid_tgt data/${dataset}/tgt-val.txt \
                     -save_data data/${dataset}/${dataset} \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
```

To train the model:

```bash
dataset=data_name
model_name=model_name
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
                 -heads 8 -transformer_ff 2048 -n_latent 0 -max_distance 1 2 3 4 \
                 -early_stopping 40 -alpha 1 \
                 -tensorboard -tensorboard_log_dir runs/${dataset}_${model_name} 2>&1 | tee train_$model_name.log


```

To test the output results:

```bash
dataset=data_name
model_name=model_name
python Generate_test_prediction.py data/${dataset}/src-test.txt -model_path experiments/${dataset}_${model_name}
python parse/parse_output.py -input_file experiments/${dataset}_${model_name}/pred/output \
                             -target_file data/${dataset}/tgt-test.txt -beam_size 10
```

To generate the shift augmented data (generate non-canonical SMILES and reordering reactants):

```bash
dataset=data_name
python data_aug.py -ori_data data/${dataset} -mode 2p2r_shift
```

Generated database will be saved to

```bash
data/${dataset}_2P2R_shift
```

Generating mask for augmented data takes 4 hours approximately.
