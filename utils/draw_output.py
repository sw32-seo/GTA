import argparse
import random
import os
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import cairosvg
import rdkit.Geometry.rdGeometry as Geo

import utils.data_utils as data_utils

import pdb


def prep_mol(smiles):
    m = Chem.MolFromSmiles(smiles)  # All smiles fed to this function should be valid
    assert m is not None

    mc = Chem.Mol(m.ToBinary())
    Chem.Kekulize(mc)

    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    return mc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', default='data/stanford_clean')
    parser.add_argument('-base_output', required=True)
    parser.add_argument('-mixture_output', required=True)
    parser.add_argument('-n_output', type=int, default=100,
                        help='Number of random output examples')
    parser.add_argument('-n_draw', type=int, default=5,
                        help='Number of output to draw per example')
    parser.add_argument('-output_dir', required=True)
    parser.add_argument('-beam_size', type=int, default=10)
    parser.add_argument('-dim', type=int, default=500)
    args = parser.parse_args()

    dim = args.dim
    n_draw = args.n_draw
    beam_size = args.beam_size
    n_output = args.n_output

    assert n_draw <= beam_size  # Cannot draw more than there are examples

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    examples_dir = '%s/examples' % output_dir
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)

    # Load data
    data_type = 'test'
    src_path = '%s/src-%s.txt' % (args.data_dir, data_type)
    tgt_path = '%s/tgt-%s.txt' % (args.data_dir, data_type)

    def parse_line_with_class_label(line):
        splits = line.strip().split(' ')
        rxn_class_label = splits[0]
        smiles_tokens = splits[1:]
        return (rxn_class_label, ''.join(smiles_tokens))

    src_data = data_utils.read_file(
        src_path, parse_func=parse_line_with_class_label)
    src_class, src_smiles = zip(*src_data)
    tgt_smiles = data_utils.read_file(tgt_path)
    print('Data loaded...')

    base_smiles = data_utils.read_file(
        args.base_output, beam_size=beam_size)
    print('Base model output smiles loaded...')
    mixture_smiles = data_utils.read_file(
        args.mixture_output, beam_size=beam_size)
    print('Mixture model output smiles loaded...')

    n_data = len(src_smiles)
    indices = list(range(n_data))

    random.shuffle(indices)
    selected_indices = indices[:n_output]

    src_class = [src_class[i] for i in selected_indices]
    src_smiles = [src_smiles[i] for i in selected_indices]
    tgt_smiles = [tgt_smiles[i] for i in selected_indices]

    base_smiles = [base_smiles[i] for i in selected_indices]
    mixture_smiles = [mixture_smiles[i] for i in selected_indices]

    output_file = open('%s/example_labels.txt' % output_dir, 'w+')
    smiles_file = open('%s/smiles.txt' % output_dir, 'w+')

    for idx in tqdm(range(n_output)):
        rxn_class = src_class[idx]

        cur_src_smiles = src_smiles[idx]
        cur_tgt_smiles = tgt_smiles[idx]

        smiles_file.write('%s,%s\n' % (cur_src_smiles, cur_tgt_smiles))

        base_smiles_beam = base_smiles[idx]
        mixture_smiles_beam = mixture_smiles[idx]

        if random.random() > 0.5:
            beam_1 = base_smiles_beam
            beam_2 = mixture_smiles_beam
            output_file.write('Example %d,%s,%s\n' % (idx, 'base', 'mixture'))
        else:
            beam_1 = mixture_smiles_beam
            beam_2 = base_smiles_beam
            output_file.write('Example %d,%s,%s\n' % (idx, 'mixture', 'base'))

        smiles_list = [cur_src_smiles, cur_tgt_smiles]

        for beam_idx in range(beam_size):
            smiles_list += [beam_1[beam_idx], beam_2[beam_idx]]

        draw_smiles_list = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                draw_smiles_list.append('')
            else:
                draw_smiles_list.append(smiles)

        draw_mols = [prep_mol(smiles) for smiles in draw_smiles_list]

        n_x, n_y = 2, args.n_draw + 1
        drawer = rdMolDraw2D.MolDraw2DSVG(n_x * dim, n_y * dim, dim, dim)
        drawer.SetFontSize(0.6)

        drawer.DrawMolecules(draw_mols)

        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        temp_path = '%s/temp' % output_dir
        f_temp = open(temp_path, 'w+')
        f_temp.write(svg)
        f_temp.close()

        cairosvg.svg2png(
            url=temp_path, write_to='%s/example_%d.png' % (examples_dir, idx))

    output_file.close()


if __name__ == '__main__':
    main()
