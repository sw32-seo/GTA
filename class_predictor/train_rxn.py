import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse

from class_predictor.models.rxn_predictor import RxnPredictor
from class_predictor.datasets.rxn_dataset import get_loader
from class_predictor.graph.mol_graph import MolGraph

from utils.data_utils import StatsTracker
import pdb


def run_epoch(model, data_loader, optimizer, training, stats, args,
              write_file=None):
    if training:
        model.train()
    else:
        model.eval()

    for batch_idx, batch_data in enumerate(tqdm(data_loader, dynamic_ncols=True)):
        if training:
            optimizer.zero_grad()

        src_smiles, tgt_smiles, class_labels = batch_data
        src_graphs = MolGraph(src_smiles, args)
        tgt_graphs = MolGraph(tgt_smiles, args)
        n_data = len(src_smiles)

        pred_logits = model(src_graphs, tgt_graphs, args)
        pred_log_probs = nn.LogSoftmax(dim=1)(pred_logits)

        class_labels = torch.tensor(class_labels, device=args.device).long()
        loss = nn.NLLLoss()(input=pred_log_probs, target=class_labels)

        stats.add_stat('loss', loss.item() * n_data, n_data)

        pred_labels = torch.argmax(pred_log_probs, dim=1)
        acc = torch.mean((pred_labels == class_labels).float())
        stats.add_stat('acc', acc.item() * n_data, n_data)

        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        if write_file is not None:
            for i in range(n_data):
                cur_src_smiles = src_smiles[i]
                cur_tgt_smiles = tgt_smiles[i]
                cur_class_label = class_labels[i].item() + 1  # Change from 0-indexed
                cur_pred_label = pred_labels[i].item() + 1

                write_file.write('%s,%s,%d,%d\n' % (
                    cur_src_smiles, cur_tgt_smiles, cur_class_label,
                    cur_pred_label))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', action='store_true', default=False,
                        help='Whether or not to use GPU.')
    parser.add_argument('-data', type=str, default='data/rxn_test',
                        help='Input data directory.')
    parser.add_argument('-output_dir', type=str, default='output/test',
                        help='The output directory.')

    parser.add_argument('-test_model', type=str, default='',
                        help='The model path used for testing')
    parser.add_argument('-test_target', type=str,
                        help='Where the test targets are located')
    parser.add_argument('-tgt_beam_size', type=int, default=1,
                        help='Number of target output for each example')
    parser.add_argument('-test_name', help='Name of test output')

    parser.add_argument('-num_epochs', type=int, default=50,
                        help='Number of epochs to train model.')
    parser.add_argument('-batch_size', type=int, default=48,
                        help='Number of examples per batch.')
    parser.add_argument('-n_classes', type=int, default=10,
                        help='Number of classes')

    parser.add_argument('-lr', type=float, default=1e-3,
                        help='The default learning rate for the optimizer.')
    parser.add_argument('-dropout', type=float, default=0.,
                        help='Dropout probability for model')
    parser.add_argument('-max_grad_norm', type=float, default=5.0,
                        help='The maximum gradient norm allowed')
    parser.add_argument('-hidden_size', type=int, default=128,
                        help='The number of hidden units for the model.')
    parser.add_argument('-depth', type=int, default=5,
                        help='The depth of the net.')

    parser.add_argument('-share_embed', action='store_true', default=False,
                        help='Whether or not to share the same conv model \
                        params for both src and tgt molecules')

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if args.cuda else 'cpu')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    rxn_predictor = RxnPredictor(args, args.n_classes)
    rxn_predictor.to(args.device)

    n_params = sum([p.nelement() for p in rxn_predictor.parameters()])
    print('N params: %d' % (n_params))

    optimizer = torch.optim.Adam(rxn_predictor.parameters(), lr=args.lr)

    if args.test_model is not '':
        rxn_predictor.load_state_dict(torch.load(args.test_model))
        print('Model loaded from %s' % args.test_model)

        src_path = '%s/src-%s.txt' % (args.data, 'test')
        test_data_loader = get_loader(
            src_path, args.test_target, batch_size=args.batch_size,
            tgt_beam_size=args.tgt_beam_size, shuffle=False)

        test_stats = StatsTracker()
        test_write_file = open('%s/preds_%s' % (args.output_dir, args.test_name), 'w+')
        with torch.no_grad():
            run_epoch(rxn_predictor, test_data_loader, None, False,
                      test_stats, args, write_file=test_write_file)
        test_write_file.close()
        test_stats.print_stats('Test Stats:')

        exit()

    dataset_loaders = {}
    for data_type in ['train', 'val', 'test']:
        src_path = '%s/src-%s.txt' % (args.data, data_type)
        tgt_path = '%s/tgt-%s.txt' % (args.data, data_type)
        dataset_loaders[data_type] = get_loader(
            src_path, tgt_path, batch_size=args.batch_size,
            shuffle=data_type == 'train')

    models_dir = '%s/models' % args.output_dir
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    preds_dir = '%s/preds' % args.output_dir
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir)

    best_epoch = 0
    best_acc = 0
    for epoch_idx in range(args.num_epochs):
        train_stats = StatsTracker()
        run_epoch(rxn_predictor, dataset_loaders['train'], optimizer, True,
                  train_stats, args)
        train_stats.print_stats('Train Epoch: %d' % epoch_idx)

        with torch.no_grad():
            dev_stats = StatsTracker()
            dev_write_file = open('%s/dev_preds_%d' % (preds_dir, epoch_idx), 'w+')
            run_epoch(rxn_predictor, dataset_loaders['val'], None, False,
                      dev_stats, args, write_file=dev_write_file)
            dev_write_file.close()
            dev_stats.print_stats('Dev Epoch: %d' % epoch_idx)

        dev_acc = dev_stats.get_stats()['acc']
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_epoch = epoch_idx
            save_path = '%s/model_%d' % (models_dir, epoch_idx)
            torch.save(rxn_predictor.state_dict(), save_path)
            print('Model saved to %s' % save_path)

    best_model_path = '%s/model_%s' % (models_dir, best_epoch)
    rxn_predictor.load_state_dict(torch.load(best_model_path))
    print('Model loaded from %s' % best_model_path)
    torch.save(rxn_predictor.state_dict(), '%s/best_model' % models_dir)

    with torch.no_grad():
        test_stats = StatsTracker()
        test_write_file = open('%s/test_preds' % preds_dir, 'w+')
        run_epoch(rxn_predictor, dataset_loaders['test'], None, False,
                  test_stats, args, write_file=test_write_file)
        test_write_file.close()
        test_stats.print_stats('Test Stats:')


if __name__ == '__main__':
    main()
