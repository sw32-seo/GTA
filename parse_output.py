import argparse
import rdkit.Chem as Chem
import numpy as np
import operator
import tqdm
import json

import pdb
import utils.data_utils as data_utils


def match_smiles_set(source_set, target_set):
    if len(source_set) != len(target_set):
        return False

    for smiles in target_set:
        if smiles not in source_set:
            return False
    return True


def match_smiles_lists(pred_list, target_list, beam_size, should_print=True):
    n_data = 0
    n_matched = np.zeros(beam_size)  # Count of matched smiles
    n_invalid = np.zeros(beam_size)  # Count of invalid smiles
    n_repeat = np.zeros(beam_size)  # Count of repeated predictions
    #
    # with open('template/rare_indices.txt', 'r+') as r_file:
    #     rare_rxn_list = json.load(r_file)

    for data_idx, target_smiles in enumerate(tqdm.tqdm(target_list)):
        # if data_idx not in rare_rxn_list:
        #     continue
        n_data += 1
        target_set = set(data_utils.canonicalize(smiles_list=target_smiles.split('.')))

        pred_beam = pred_list[data_idx]

        beam_matched = False
        prev_sets = []
        for beam_idx, pred_smiles in enumerate(pred_beam):
            pred_set = set(data_utils.canonicalize(smiles_list=pred_smiles.split('.')))
            if '' in pred_set:
                pred_set.remove('')
            set_matched = match_smiles_set(pred_set, target_set)

            # Check if current pred_set matches any previous sets
            for prev_set in prev_sets:
                if match_smiles_set(pred_set, prev_set):
                    n_repeat[beam_idx] += 1

            if len(pred_set) > 0:
                # Add pred set to list of predictions for current example
                prev_sets.append(pred_set)
            else:
                # If the pred set is empty and the string is not, then invalid
                if pred_smiles != '':
                    n_invalid[beam_idx] += 1

            # Increment if not yet matched beam and the pred set matches
            if set_matched and not beam_matched:
                n_matched[beam_idx] += 1
                beam_matched = True

    if should_print:
        print('total examples: %d' % n_data)
        for beam_idx in range(beam_size):
            match_perc = np.sum(n_matched[:beam_idx+1]) / n_data
            invalid_perc = n_invalid[beam_idx] / n_data
            repeat_perc = n_repeat[beam_idx] / n_data

            print('beam: %d, matched: %.3f, invalid: %.3f, repeat: %.3f' %
                (beam_idx+1, match_perc, invalid_perc, repeat_perc))

    return n_data, n_matched, n_invalid, n_repeat


def combine_latent(input_dir, n_latent, beam_size, output_path=None, clean=False):
    """
    Reads the output smiles from each of the latent classes and combines them.

    Args:
        input_dir: The path to the input directory containing output files
        n_latent: The number of latent classes used for the model
        beam_size: Number of smiles results per reaction
        output_path: If given, writes the combined smiles to this path
    """
    # results_path is the prefix for the different latent file outputs
    latent_list = []

    def parse(line):
        c_line = line.strip().replace(' ', '')
        smiles, score = c_line.split(',')
        score = float(score)
        return (smiles, score)

    for latent_idx in range(n_latent):
        file_path = '%s/output_%d' % (input_dir, latent_idx)
        smiles_list = data_utils.read_file(file_path, beam_size=beam_size,
                                           parse_func=parse)
        latent_list.append(smiles_list)

    combined_list = []

    if output_path is not None:
        output_file = open(output_path, 'w+')

    n_data = len(latent_list[0])
    for data_idx in tqdm.tqdm(range(n_data)):
        r_dict = {}
        for latent_idx in range(n_latent):
            output_list = latent_list[latent_idx][data_idx]
            for smiles, score in output_list:

                if clean:
                    smiles = data_utils.canonicalize(smiles)
                    if smiles == '':
                        continue

                if smiles not in r_dict:  # Add the output to dictionary
                    r_dict[smiles] = (score, latent_idx)
                else:
                    if score > r_dict[smiles][0]:  # Update with the best score if applicable
                        r_dict[smiles] = (score, latent_idx)
        sorted_output = sorted(r_dict.items(), key=operator.itemgetter(1),
                               reverse=True)
        top_smiles = []
        for beam_idx in range(beam_size):
            if beam_idx < len(sorted_output):
                smiles, (score, latent_idx) = sorted_output[beam_idx]
                top_smiles.append(smiles)

                if output_path is not None:
                    output_file.write('%s,%.4f,%d\n' % (smiles, score, latent_idx))

        combined_list.append(top_smiles)
    if output_path is not None:
        output_file.close()
    return combined_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir', type=str)
    parser.add_argument('-input_file', type=str, default='',
                        help='Optional single input file input')
    parser.add_argument('-target_file', type=str, required=True)
    parser.add_argument('-n_latent', type=int, default=0)
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-clean', action='store_true', default=False)
    args = parser.parse_args()

    beam_size, n_latent = args.beam_size, args.n_latent

    if n_latent > 1:
        smiles_list = combine_latent(
            input_dir=args.input_dir, n_latent=n_latent, beam_size=beam_size,
            output_path='%s/combined' % args.input_dir, clean=args.clean)
    else:
        smiles_list = data_utils.read_file(args.input_file, beam_size=beam_size)

    target_list = data_utils.read_file(args.target_file, beam_size=1)
    match_smiles_lists(smiles_list, target_list, beam_size)


if __name__ == '__main__':
    main()
