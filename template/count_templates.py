import argparse
import re
import os
import random
import json
from tqdm import tqdm
import math
import rdkit.Chem as Chem
import numpy as np

import utils.data_utils as data_utils

from template.generate_retro_templates import process_an_example
from template.rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants

import pdb


def unmapped_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    return Chem.MolToSmiles(mol)


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-input_data', default='template/data_processed.csv')
    # args = parser.parse_args()
    #
    # with open(args.input_data, 'r+') as data_file:
    #     data = []
    #     skip_header = True
    #     for line in data_file.readlines():
    #         if skip_header:
    #             skip_header = False
    #             continue
    #         splits = line.strip().split(',')
    #         rxn_smiles = splits[4]
    #         data.append(rxn_smiles)
    #
    # # Split all the templates from the data into train and test_templates
    # template_dict = {}
    # for idx, rxn_smiles in enumerate(tqdm(data)):
    #     template = process_an_example(rxn_smiles, super_general=True)
    #     if template is None:
    #         continue
    #     template = '({})>>{}'.format(template.split('>>')[0], template.split('>>')[1])
    #
    #     if template in template_dict:
    #         template_dict[template] += 1
    #     else:
    #         template_dict[template] = 1

    # counts = np.array(list(template_dict.values()))

    with open('template/all_template_counts.json', 'r+') as template_file:
        template_dict = json.load(template_file)

    data_dir = 'data/stanford_no_rxn'
    test_src_data, test_tgt_data = data_utils.read_src_tgt_files(data_dir, 'test')

    test_mapped_data = {}
    with open('template/data_processed.csv', 'r+') as data_file:
        data = []
        skip_header = True
        for line in tqdm(data_file.readlines()):
            if skip_header:
                skip_header = False
                continue
            splits = line.strip().split(',')
            rxn_smiles = splits[4]
            data.append(rxn_smiles)

            tgt_smiles, src_smiles = rxn_smiles.split('>>')
            unmapped_src_smiles = unmapped_smiles(src_smiles)
            unmapped_tgt_smiles = unmapped_smiles(tgt_smiles)

            for idx, test_src_smiles in enumerate(test_src_data):
                if unmapped_src_smiles == test_src_smiles:
                    if unmapped_tgt_smiles == test_tgt_data[idx]:
                        test_mapped_data[idx] = rxn_smiles

    rare_rxn_indices = []
    for idx, rxn_smiles in tqdm(test_mapped_data.items()):
        template = process_an_example(rxn_smiles, super_general=True)
        if template is not None:
            template = '({})>>{}'.format(template.split('>>')[0], template.split('>>')[1])
        if template in template_dict:
            if template_dict[template] <= 10:
                rare_rxn_indices.append(idx)

    with open('template/rare_indices.txt', 'w+') as out_file:
        json.dump(rare_rxn_indices, out_file)

    pdb.set_trace()


if __name__ == '__main__':
    main()
