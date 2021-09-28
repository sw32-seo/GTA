import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits.axes_grid1 as axes_grid1
import torch

import pdb

class_norm = np.array([0.30221636, 0.23808382, 0.1126966 , 0.01798669, 0.01299039, 0.16693647, 0.09163219, 0.01626796, 0.03665288, 0.00453664])


def read_latent_class(file_path, beam_size):
    latent_preds = []
    with open(file_path, 'r+') as file:
        current_beam = []
        for line in file.readlines():
            latent_idx = int(line.strip().split(',')[-1])
            current_beam.append(latent_idx)

            if len(current_beam) == beam_size:
                latent_preds.append(current_beam)
                current_beam = []
    return latent_preds


def read_preds(pred_path, beam_size):
    all_preds = []
    with open(pred_path, 'r+') as pred_file:
        current_beam = []
        for line in pred_file.readlines():
            splits = line.strip().split(',')
            src_smiles = splits[0]
            tgt_smiles = splits[1]
            class_label = int(splits[2])
            pred_label = int(splits[3])

            current_beam.append((src_smiles, tgt_smiles, class_label, pred_label))
            if len(current_beam) == beam_size:
                all_preds.append(current_beam)
                current_beam = []
    return all_preds


def diversity_measure(all_preds):
    # Measures the average number of class representation in each beam
    counts = []
    for pred_beam in all_preds:
        beam_classes = []
        for idx, (src_smiles, tgt_smiles, class_label, pred_label) in enumerate(pred_beam):
            if tgt_smiles == '':
                # Invalid
                continue
            else:
                if pred_label not in beam_classes:
                    beam_classes.append(pred_label)
        counts.append(len(beam_classes))
    counts = np.array(counts)
    mean, std = np.mean(counts), np.std(counts)
    return mean, std


def get_pred_counts(all_preds, n_classes):
    counts = np.zeros([n_classes])
    for pred_beam in all_preds:
        for idx, (src_smiles, tgt_smiles, class_label, pred_label) in enumerate(pred_beam):
            if idx >= 5:
                break
            if tgt_smiles == '':
                continue
            else:
                counts[pred_label-1] += 1
    return counts


def get_rxn_counts(all_preds, all_latents, n_classes, n_latent):
    rxn_counts = np.zeros([n_classes, n_latent])
    for ex_idx, pred_beam in enumerate(all_preds):
        for beam_idx, (src_smiles, tgt_smiles, class_label, pred_label) in enumerate(pred_beam):
            latent_idx = all_latents[ex_idx][beam_idx]
            rxn_counts[class_label-1][latent_idx] += 1
    return rxn_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred_file', default='')
    parser.add_argument('-latent_file', default='')
    parser.add_argument('-beam_size', type=int, default=1)
    parser.add_argument('-n_classes', type=int, default=10)
    parser.add_argument('-mode', choices=['diversity', 'pred_dist', 'rxn_dist'], required=True)
    parser.add_argument('-n_latent', type=int, default=1)
    args = parser.parse_args()

    if args.mode == 'diversity':
        all_preds = read_preds(args.pred_file, args.beam_size)
        mean, std = diversity_measure(all_preds)
        print('Mean: %.3f, Std: %.3f' % (mean, std))
    elif args.mode == 'pred_dist':
        all_counts = []
        for latent_idx in range(args.n_latent):
            print('Reading latent output: %d' % latent_idx)
            pred_path = '%s_%d' % (args.pred_file, latent_idx)
            all_preds = read_preds(pred_path, args.beam_size)
            counts = get_pred_counts(all_preds, args.n_classes)
            all_counts.append(counts)
        all_counts = np.stack(all_counts)
        norm_counts = all_counts / np.expand_dims(class_norm, axis=0)
        sum_counts = np.sum(all_counts, axis=0, keepdims=True)
        probs = all_counts / sum_counts

        # fig = plt.figure()
        # grid = axes_grid1.AxesGrid(
        #     fig, 111, nrows_ncols=(1, 2), axes_pad = 0.5, cbar_location = "right",
        #     cbar_mode="each", cbar_size="15%", cbar_pad="5%",)

        im0 = plt.imshow(probs, cmap='Blues', interpolation='nearest')
        # grid.cbar_axes[0].colorbar(im0)
        plt.xlabel('Reaction Class')
        plt.ylabel('Latent Class')
        plt.savefig('output/heat_map.png')
        pdb.set_trace()
    elif args.mode == 'rxn_dist':
        all_preds = read_preds(args.pred_file, args.beam_size)
        all_latents = read_latent_class(args.latent_file, args.beam_size)
        rxn_counts = get_rxn_counts(all_preds, all_latents, args.n_classes, args.n_latent)
        sum_counts = np.sum(rxn_counts, axis=1, keepdims=True)
        norm_counts = rxn_counts / sum_counts

        norm = mpl.colors.Normalize(vmin=0.5, vmax=1.)

        plt.imshow(norm_counts, norm=norm, cmap='Blues', interpolation='nearest')
        plt.ylabel('Reaction Class')
        plt.xlabel('Latent Class')
        plt.savefig('output/rxn_heat_map.png')
        pdb.set_trace()
    else:
        assert False



if __name__ == '__main__':
    main()
