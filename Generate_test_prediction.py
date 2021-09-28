import subprocess
import os
import argparse

parser = argparse.ArgumentParser(description="Get saved data/model path")
parser.add_argument('--src', '-src', type=str, default='data/USPTO-50k_no_rxn/src-test.txt')
parser.add_argument('--model_path', '-model_path', type=str)
args = parser.parse_args()

src = args.src
models_path = os.path.join(args.model_path, 'models')

a = os.listdir(models_path)
a.sort()
best_model_path = os.path.join(models_path, a[-1])
subprocess.call(['python', 'translate.py',
                 '-gpu', '0',
                 '-model', '%s' % best_model_path,
                 '-src', '%s' % src,
                 '-output_dir', '%s/pred' % args.model_path,
                 '-batch_size', '32',
                 '-replace_unk',
                 '-max_length', '256',
                 '-n_best', '50',
                 '-beam_size', '10',
                 '-log_probs',
                 '-n_translate_latent', '0'
                 ])
