import argparse
import torch
import os
from tqdm import tqdm


from class_predictor.models.rxn_predictor import RxnPredictor
from class_predictor.datasets.rxn_dataset import get_loader
from class_predictor.train_rxn import run_epoch
from utils.data_utils import StatsTracker
import pdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', action='store_true', default=False,
                        help='Whether or not to use GPU.')
    parser.add_argument('-output_dir', type=str, default='output/',
                        help='The output directory.')
    parser.add_argument('-test_name', default='',
                        help='Name of test output')

    parser.add_argument('-src_file', type=str, required=True)
    parser.add_argument('-tgt_file', type=str, required=True)
    parser.add_argument('-n_tgt', type=int, default=1)

    parser.add_argument('-test_model', type=str, required=True,
                        help='The model path used for testing')
    parser.add_argument('-tgt_beam_size', type=int, default=1,
                        help='Number of target output for each example')

    parser.add_argument('-n_classes', type=int, default=10,
                        help='Number of classes')

    parser.add_argument('-batch_size', type=int, default=48,
                        help='Number of examples per batch.')
    parser.add_argument('-dropout', type=float, default=0.,
                        help='Dropout probability for model')
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

    rxn_predictor.load_state_dict(torch.load(args.test_model))

    if args.n_tgt == 1:
        test_data_loader = get_loader(
            args.src_file, args.tgt_file, batch_size=args.batch_size,
            tgt_beam_size=args.tgt_beam_size, shuffle=False)

        test_stats = StatsTracker()
        test_write_file = open('%s/preds_%s' % (args.output_dir, args.test_name), 'w+')
        with torch.no_grad():
            run_epoch(rxn_predictor, test_data_loader, None, False,
                      test_stats, args, write_file=test_write_file)
        test_write_file.close()
        test_stats.print_stats('Test Stats:')
    else:
        for i in range(args.n_tgt):
            print('Evaluating latent idx: %d' % (i))
            cur_tgt_file = '%s_%d' % (args.tgt_file, i)
            test_data_loader = get_loader(
                args.src_file, cur_tgt_file, batch_size=args.batch_size,
                tgt_beam_size=args.tgt_beam_size, shuffle=False)

            test_stats = StatsTracker()
            test_write_file = open('%s/preds_%s_%d' % (
                args.output_dir, args.test_name, i), 'w+')
            with torch.no_grad():
                run_epoch(rxn_predictor, test_data_loader, None, False,
                          test_stats, args, write_file=test_write_file)
            test_write_file.close()
            test_stats.print_stats('Test Stats:')


if __name__ == '__main__':
    main()
