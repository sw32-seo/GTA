import argparse
import os
import rdkit.Chem as Chem
from tqdm import tqdm
import json
import random
import sys

import utils.data_utils as data_utils
sys.path.insert(0, './template')
from template.rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants

import pdb

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

MAX_ITS = 100


SYNTAX_TOKENS = ['(', ')', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ATOM_TOKENS = ['[O-]', '[NH+]', '[C@H]', '[Br-]', '[Cl+3]', '[PH4]', '[Pt]', 'O', '[S+]', '[C@@]', '[B-]', 'F', 'S', 'N', '[S@@]', '<MASK>', '[Pd]', '[NH4+]', '[Se]', '[Si]', '[NH3+]', '[P+]', '[Cu]', 'C', '[N+]', '[SH]', '[SiH2]', '[n-]', '<blank>', '[n+]', '</s>', '<s>', '[Zn+]', '[nH]', '[OH-]', '[PH]', 'o', '[N@+]', '[BH3-]', '<unk>', '[N-]', '[Cl-]', 'B', '[S@]', '[PH2]', '[SiH]', '[s+]', '[C@@H]', '[Sn]', 'c', 'Br', '.', 's', 'P', '[SnH]', '[I+]', 'n', '[Zn]', '[Fe]', 'I', '[BH-]', '[Li]', '[NH2+]', '[C@]', '[S-]', '[NH-]', '[Mg]', '[K]', '[C-]', 'Cl', '[Mg+]', '[se]']
BOND_TOKENS = ['-', '=', '#', '/', '\\']
MASK_TOKEN = '<MASK>'


def write_smiles(file, smiles_list):
    for smiles in smiles_list:
        tokens_string = data_utils.smi_tokenizer(smiles)
        tokens = tokens_string.split(' ')

        tokens_string = ' '.join(tokens)
        file.write(tokens_string + '\n')


def get_random_set(smiles):
    # Given a smiles string, break a random single bond as two
    mol = Chem.MolFromSmiles(smiles)

    n_bonds = mol.GetNumBonds()
    if n_bonds == 0:
        return None
    n = 0
    while True:
        n += 1

        if n > MAX_ITS:
            return None

        rand_bond = mol.GetBonds()[random.randint(0, n_bonds-1)]

        if rand_bond.GetIsAromatic():
            continue

        if rand_bond.IsInRing():
            continue

        bond_type = rand_bond.GetBondType()
        if bond_type != Chem.rdchem.BondType.SINGLE:
            continue

        new_mol = Chem.rdmolops.FragmentOnBonds(mol, [rand_bond.GetIdx()], addDummies=False)

        try:
            new_smiles = Chem.MolToSmiles(new_mol)
            sanitize_mol = Chem.MolFromSmiles(new_smiles)
            if sanitize_mol is None:
                continue
            return Chem.MolToSmiles(sanitize_mol)
        except:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', required=True)
    parser.add_argument('-output_dir', required=True)
    parser.add_argument('-template', type=str)
    parser.add_argument('-n_aug', default=5, type=int)
    parser.add_argument('-type', choices=['sets', 'templates'], required=True)
    parser.add_argument('-transductive', action='store_true', default=False)
    args = parser.parse_args()

    # Make output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Read the source SMILES
    src_smiles_dict = {}
    for data_type in ['train', 'val', 'test']:
        src_smiles_list = data_utils.read_file(
            '%s/src-%s.txt' % (args.data_dir, data_type))
        src_smiles_dict[data_type] = src_smiles_list

    if args.type == 'templates':
        with open(args.template, 'r+') as template_file:
            template_dict = json.load(template_file)
            template_list = list(template_dict.keys())

    # If transductive flag is on, add the valid and test source smiles into the train set
    if args.transductive:
        src_smiles_dict['train'] = src_smiles_dict['train'] + src_smiles_dict['val'] + src_smiles_dict['test']

    for data_type in ['train', 'val', 'test']:
        output_src = open('%s/src-%s.txt' % (args.output_dir, data_type), 'w+')
        output_tgt = open('%s/tgt-%s.txt' % (args.output_dir, data_type), 'w+')

        src_smiles_list = src_smiles_dict[data_type]
        for smiles in tqdm(src_smiles_list):
            if args.type == 'sets':
                new_smiles_list = []
                n = 0
                while len(new_smiles_list) < args.n_aug:
                    n += 1
                    if n > MAX_ITS:
                        break
                    new_smiles = get_random_set(smiles)
                    if new_smiles is not None and new_smiles not in new_smiles_list:
                        new_smiles_list.append(new_smiles)
                write_smiles(output_src, [smiles] * len(new_smiles_list))
                write_smiles(output_tgt, new_smiles_list)
            elif args.type == 'templates':
                # randomly shuffle the template list
                random.shuffle(template_list)
                new_smiles_list = []

                for template in template_list:
                    rd_rxn = rdchiralReaction(template)
                    rd_rct = rdchiralReactants(smiles)

                    outcome_list = rdchiralRun(rd_rxn, rd_rct)
                    if len(outcome_list) > 0:
                        random.shuffle(outcome_list)
                        new_smiles = outcome_list[0]
                        if new_smiles not in new_smiles_list:
                            new_smiles_list.append(new_smiles)
                    if len(new_smiles_list) >= args.n_aug:
                        break
                write_smiles(output_src, [smiles] * len(new_smiles_list))
                write_smiles(output_tgt, new_smiles_list)


if __name__ == '__main__':
    main()
