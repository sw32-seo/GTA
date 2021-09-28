# Test to make sure nothing in test can be constructed from training examples
import rdkit.Chem as Chem
from tqdm import tqdm
import json

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


def match_smiles(source_smiles, target_smiles):
    # Returns true if given src and tgt smiles are equivalent sets
    source_set = set(source_smiles.split('.'))
    target_set = set(target_smiles.split('.'))

    if len(source_set) != len(target_set):
        return False

    for smiles in target_set:
        if smiles not in source_set:
            return False
    return True


def main():
    data_dir = 'data/template_data'
    test_src_data, test_tgt_data = data_utils.read_src_tgt_files(
        data_dir=data_dir, data_type='test')

    with open('%s/train_templates.json' % data_dir, 'r+') as template_file:
        template_list = json.load(template_file)

    n_matched = 0
    for idx, src_smiles in enumerate(tqdm(test_src_data)):
        tgt_smiles = test_tgt_data[idx]

        rd_rct = rdchiralReactants(src_smiles)
        all_outcomes = set()
        for template in template_list:
            rd_rxn = rdchiralReaction(template)
            outcomes = rdchiralRun(rd_rxn, rd_rct, combine_enantiomers=False)

            matched = False
            for outcome_smiles in outcomes:
                all_outcomes.add(outcome_smiles)
                matched = match_smiles(source_smiles=outcome_smiles,
                                       target_smiles=tgt_smiles)
                if matched:
                    break

            if matched:
                n_matched += 1
                break
        all_outcomes_list = list(all_outcomes)

    print('N matched: %d' % (n_matched))
    pdb.set_trace()



if __name__ == '__main__':
    main()
