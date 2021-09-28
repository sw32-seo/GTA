import os
import argparse
import rdkit.Chem as Chem
import tqdm

import utils.data_utils as data_utils
import pdb

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')


def augment_data(src_data, tgt_data, output_dir, data_type, n_aug):
    output_src = open('%s/src-%s.txt' % (output_dir, data_type), 'w+')
    output_tgt = open('%s/tgt-%s.txt' % (output_dir, data_type), 'w+')

    rxn_data = list(zip(src_data, tgt_data))

    for src_smiles, tgt_smiles in tqdm.tqdm(rxn_data):
        src_smiles = data_utils.canonicalize_smiles(src_smiles)
        tgt_smiles = data_utils.canonicalize_smiles(tgt_smiles)

        output_rxns = [(src_smiles, tgt_smiles)]
        tgt_set = set(tgt_smiles.split('.'))
        while len(output_rxns) < n_aug + 1:
            try:
                new_tgt_smiles = data_utils.canonicalize_smiles(Chem.MolToSmiles(
                    Chem.MolFromSmiles(tgt_smiles), doRandom=True))

                # Sanity check:
                new_set = set(new_tgt_smiles.split('.'))
                if not data_utils.match_smiles_set(tgt_set, new_set):
                    continue
                output_rxns.append((src_smiles, new_tgt_smiles))
            except:
                pass
        for src_smiles, tgt_smiles in output_rxns:
            output_src.write('%s\n' % data_utils.smi_tokenizer(src_smiles))
            output_tgt.write('%s\n' % data_utils.smi_tokenizer(tgt_smiles))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', required=True)
    parser.add_argument('-output_dir', required=True)
    parser.add_argument('-n_aug', type=int, default=0)
    parser.add_argument('-remove_rxn', action='store_true', default=False)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    def read_rxn_line(line):
        tokens = line.strip().split(' ')
        if args.remove_rxn:
            tokens = tokens[1:]
        return ''.join(tokens)

    src_data, tgt_data = data_utils.read_src_tgt_files(
        data_dir=args.data_dir, data_type='train', source_func=read_rxn_line)
    augment_data(src_data, tgt_data, args.output_dir, 'train', args.n_aug)

    src_data, tgt_data = data_utils.read_src_tgt_files(
        data_dir=args.data_dir, data_type='val', source_func=read_rxn_line)
    augment_data(src_data, tgt_data, args.output_dir, 'val', 0)

    src_data, tgt_data = data_utils.read_src_tgt_files(
        data_dir=args.data_dir, data_type='test', source_func=read_rxn_line)
    augment_data(src_data, tgt_data, args.output_dir, 'test', 0)


if __name__ == '__main__':
    main()
