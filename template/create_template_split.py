import argparse
import re
import os
import random
import json
from tqdm import tqdm
import math
import rdkit.Chem as Chem

from template.generate_retro_templates import process_an_example
from template.rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants

import pdb


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def unmapped_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    return Chem.MolToSmiles(mol)


def write_rxn_smiles(src_file, tgt_file, rxn_smiles, rxn_class=None):
    r_smiles, p_smiles = rxn_smiles.split('>>')
    r_mol = Chem.MolFromSmiles(r_smiles)
    p_mol = Chem.MolFromSmiles(p_smiles)

    def remove_mapping(mol):
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')

    remove_mapping(r_mol)
    remove_mapping(p_mol)

    tgt_tokens = smi_tokenizer(Chem.MolToSmiles(r_mol))
    src_tokens = smi_tokenizer(Chem.MolToSmiles(p_mol))

    if rxn_class is not None:
        src_tokens = ('<RX_%d> ' % rxn_class) + src_tokens
    src_file.write(src_tokens + '\n')
    tgt_file.write(tgt_tokens + '\n')


def match_smiles(source_smiles, target_smiles):
    source_set = set(source_smiles.split('.'))
    target_set = set(target_smiles.split('.'))

    if len(source_set) != len(target_set):
        return False

    for smiles in target_set:
        if smiles not in source_set:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-template_path', default='data/template_data_2/all_templates.json')
    parser.add_argument('-input_data', default='template/data_processed.csv')
    parser.add_argument('-output_dir', default='data/template_data_2')
    parser.add_argument('-template_frac', type=float, default=0.825)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # hard-coded data file
    with open(args.input_data, 'r+') as data_file:
        data = []
        skip_header = True
        for line in data_file.readlines():
            if skip_header:
                skip_header = False
                continue
            splits = line.strip().split(',')
            rxn_smiles = splits[4]
            data.append(rxn_smiles)

    print('Raw data read...')

    # # Split all the templates from the data into train and test_templates
    # all_templates = []
    # for idx, rxn_smiles in enumerate(tqdm(data)):
    #     template = process_an_example(rxn_smiles, super_general=True)
    #     if template is None:
    #         continue
    #     template = '({})>>{}'.format(template.split('>>')[0], template.split('>>')[1])
    #
    #     if template in all_templates:
    #         continue
    #
    #     all_templates.append(template)
    #
    # with open('%s/all_templates.json' % args.output_dir, 'w+') as template_file:
    #     json.dump(all_templates, template_file)

    with open(args.template_path, 'r+') as template_file:
        template_list = json.load(template_file)

    random.shuffle(template_list)

    n_templates = len(template_list)
    n_train = math.ceil(n_templates * args.template_frac)
    train_templates = template_list[:n_train]
    test_templates = template_list[n_train:]

    print('N train templates: %d, N test templates: %d' % (
        len(train_templates), len(test_templates)))

    with open('%s/train_templates.json' % args.output_dir, 'w+') as train_template_file:
        json.dump(train_templates, train_template_file)
    with open('%s/test_templates.json' % args.output_dir, 'w+') as test_template_file:
        json.dump(test_templates, test_template_file)

    train_rxns, test_rxns = [], []
    n_skipped = 0
    for idx, rxn_smiles in enumerate(tqdm(data)):
        if (idx + 1) % 100 == 0:
            print('N train rxns: %d, N test rxns: %d' % (len(train_rxns), len(test_rxns)))
        tgt_smiles, src_smiles = rxn_smiles.split('>>')

        unmapped_src_smiles = unmapped_smiles(src_smiles)
        unmapped_tgt_smiles = unmapped_smiles(tgt_smiles)
        rd_rct = rdchiralReactants(unmapped_src_smiles)

        template = process_an_example(rxn_smiles, super_general=True)
        if template is not None:
            template = '({})>>{}'.format(template.split('>>')[0], template.split('>>')[1])
        if template in train_templates:
            train_rxns.append((unmapped_src_smiles, unmapped_tgt_smiles))
            continue

        for template in train_templates:
            rd_rxn = rdchiralReaction(template)

            outcomes = rdchiralRun(rd_rxn, rd_rct, combine_enantiomers=False)
            matched = False
            for outcome_smiles in outcomes:
                matched = match_smiles(source_smiles=outcome_smiles, target_smiles=unmapped_tgt_smiles)
                if matched:
                    break
            if matched:
                break
        if matched:
            n_skipped += 1
        else:
            test_rxns.append((unmapped_src_smiles, unmapped_tgt_smiles))

    n_train = len(train_rxns)
    n_val = math.ceil(n_train * 0.1)

    random.shuffle(train_rxns)
    val_rxns = train_rxns[:n_val]
    train_rxns = train_rxns[n_val:]

    outputs = {}
    for type in ['train', 'val', 'test']:
        for loc in ['src', 'tgt']:
            outputs['%s-%s' % (loc, type)] = open(
                '%s/%s-%s.txt' % (args.output_dir, loc, type), 'w+')

    for src_smiles, tgt_smiles in train_rxns:
        outputs['src-train'].write(smi_tokenizer(src_smiles) + '\n')
        outputs['tgt-train'].write(smi_tokenizer(tgt_smiles) + '\n')

    for src_smiles, tgt_smiles in val_rxns:
        outputs['src-val'].write(smi_tokenizer(src_smiles) + '\n')
        outputs['tgt-val'].write(smi_tokenizer(tgt_smiles) + '\n')

    for src_smiles, tgt_smiles in test_rxns:
        outputs['src-test'].write(smi_tokenizer(src_smiles) + '\n')
        outputs['tgt-test'].write(smi_tokenizer(tgt_smiles) + '\n')

    pdb.set_trace()


if __name__ == '__main__':
    main()
