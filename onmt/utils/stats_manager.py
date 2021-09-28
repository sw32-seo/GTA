import numpy as np


class StatsManager(object):
    def __init__(self, stat_names=['step', 'acc', 'ppl']):
        self.stat_names = stat_names
        self.train_stats = {}
        self.val_stats = {}

        for name in stat_names:
            self.train_stats[name] = []
            self.val_stats[name] = []

    def add_stats(self, train_stats=None, valid_stats=None):
        assert train_stats is not None or valid_stats is not None

        if train_stats is not None:
            for name, val in train_stats.items():
                self.train_stats[name].append(val)
            return

        if valid_stats is not None:
            for name, val in valid_stats.items():
                self.val_stats[name].append(val)

    def get_best_model(self, stat_name='acc', highest_best=True):
        stat_list = np.array(self.val_stats[stat_name])[10:]

        if highest_best:
            best_idx = np.argmax(stat_list)
        else:
            best_idx = np.argmin(stat_list)

        best_stats = {}
        for name in self.stat_names:
            best_stats[name] = self.val_stats[name][10:][best_idx]

        return self.val_stats['step'][10:][best_idx], best_stats

    def write_stats(self, output_dir):
        with open('%s/train_stats.csv' % output_dir, 'w+') as train_file:
            steps = self.train_stats['step']
            for idx, step in enumerate(steps):
                acc = self.train_stats['acc'][idx]
                ppl = self.train_stats['ppl'][idx]

                train_file.write('%s,%.4f,%.4f\n' % (step, acc, ppl))

        with open('%s/valid_stats.csv' % output_dir, 'w+') as valid_file:
            steps = self.val_stats['step']
            for idx, step in enumerate(steps):
                acc = self.val_stats['acc'][idx]
                ppl = self.val_stats['ppl'][idx]

                valid_file.write('%s,%.4f,%.4f\n' % (step, acc, ppl))
