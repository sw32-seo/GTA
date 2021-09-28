# Read in the set of templates, and filter out templates not present in the train set

import json
from tqdm import tqdm

import utils.data_utils as data_utils
from template.rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants

import pdb


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
    data_dir = 'data/stanford_no_rxn'
    template_json_path = 'template/templates_raw.json'
    template_output_path = 'template/templates_train.json'

    with open(template_json_path, 'r+') as template_file:
        template_dict = json.load(template_file)
        template_list = list(template_dict.keys())
    print('Templates read...')

    src_rxn_list, tgt_rxn_list = data_utils.read_src_tgt_files(data_dir, 'train')
    print('Reactions read...')

    rxn_list = list(zip(src_rxn_list, tgt_rxn_list))

    n_templates_skipped = 0
    template_output_dict = {}
    for template_idx, template in enumerate(tqdm(template_list)):
        rd_rxn = rdchiralReaction(template)

        template_matched = False
        for rxn_idx, (src_smiles, tgt_smiles) in enumerate(rxn_list):
            rd_rct = rdchiralReactants(src_smiles)

            rxn_matched = False
            outcome_list = rdchiralRun(rd_rxn, rd_rct, combine_enantiomers=False)
            for outcome_smiles in outcome_list:
                rxn_matched = match_smiles(source_smiles=outcome_smiles,
                                           target_smiles=tgt_smiles)
                if rxn_matched:
                    break

            if rxn_matched:
                template_matched = True
                break
        if template_matched:
            template_output_dict[template] = template_dict[template]
        else:
            n_templates_skipped += 1

    print('Number of templates skipped: %d' % n_templates_skipped)

    with open(template_output_path, 'w+') as template_output_file:
        json.dump(template_output_dict, template_output_file)

    pdb.set_trace()


if __name__ == '__main__':
    main()
