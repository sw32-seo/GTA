import json
import random
import pdb
import rdkit.Chem as Chem
import numpy as np
from tqdm import tqdm

import utils.data_utils as data_utils
from template.rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants


def main():
    with open('template/templates_train.json', 'r+') as template_file:
        template_list = json.load(template_file)
        template_list = list(template_list.keys())

    data_dir = 'data/stanford_no_rxn'
    train_src_data, _ = data_utils.read_src_tgt_files(
        data_dir=data_dir, data_type='train')

    random.shuffle(train_src_data)
    train_src_data = train_src_data[:500]

    n_r_counts = []
    n_t_counts = []
    for train_idx, train_smiles in enumerate(tqdm(train_src_data)):
        mol = Chem.MolFromSmiles(train_smiles)

        n_possible_bonds = 0
        for bond in mol.GetBonds():
            if not bond.IsInRing() and not bond.GetIsAromatic():
                bond_type = bond.GetBondType()
                if bond_type == Chem.rdchem.BondType.SINGLE:
                    n_possible_bonds += 1
        n_r_counts.append(n_possible_bonds)

        possible_temps = set()
        rd_rct = rdchiralReactants(train_smiles)
        for template in template_list:
            rd_rxn = rdchiralReaction(template)

            outcome_list = rdchiralRun(rd_rxn, rd_rct)
            for outcome_smiles in outcome_list:
                possible_temps.add(outcome_smiles)
        n_t_counts.append(len(possible_temps))


    n_r_counts = np.array(n_r_counts)
    n_t_counts = np.array(n_t_counts)

    pdb.set_trace()
    # for template in


if __name__ == '__main__':
    main()
