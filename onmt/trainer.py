"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from copy import deepcopy
import torch
import numpy as np
import traceback

import onmt.utils
from onmt.utils.logging import logger
import onmt.utils.latent_utils as latent_utils
import rdkit.Chem as Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix, GetDistanceMatrix

import pdb


def build_trainer(opt, device_id, model, fields, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    tgt_field = dict(fields)["tgt"].base_field
    train_loss = onmt.utils.loss.build_loss_compute(model, tgt_field, opt)
    latent_loss = None
    if model.n_latent > 1:
        latent_loss = onmt.utils.loss.build_loss_compute(
            model, tgt_field, opt, train=False, reduce=False)
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, tgt_field, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    report_manager = onmt.utils.build_report_manager(opt)
    trainer = onmt.Trainer(model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver if gpu_rank == 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper,
                           dropout=dropout,
                           dropout_steps=dropout_steps,
                           latent_loss=latent_loss,
                           segment_token_idx=opt.segment_token_idx,
                           stoi=tgt_field.vocab.stoi,
                           max_dist=opt.max_distance,
                           alpha=opt.alpha)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0],
                 latent_loss=None, segment_token_idx=None, stoi=None, max_dist=None,
                 alpha=1.0):
        # Basic attributes.
        self.model = model
        self.n_latent = model.n_latent
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.latent_loss = latent_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.segment_token_idx = segment_token_idx
        self.stoi = stoi
        self.inv_stoi = {v: k for k, v in stoi.items()}
        self.max_dist = max_dist
        self.alpha = alpha

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d"
                            % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1)/(step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = onmt.utils.Statistics(n_latent=self.n_latent)
        report_stats = onmt.utils.Statistics(n_latent=self.n_latent)
        self._start_report_manager(start_time=total_stats.start_time)

        stats_manager = onmt.utils.StatsManager()

        for i, (batches, normalization) in enumerate(
                self._accum_batches(train_iter)):
            step = self.optim.training_step
            # UPDATE DROPOUT
            self._maybe_update_dropout(step)

            if self.gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d \
                            n_minibatch %d"
                            % (self.gpu_rank, i + 1, len(batches)))

            if self.n_gpu > 1:
                normalization = sum(onmt.utils.distributed
                                    .all_gather_list
                                    (normalization))

            self._gradient_accumulation(
                batches, normalization, total_stats,
                report_stats)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            if step % self.report_manager.report_every == 0:
                stats_manager.add_stats(
                    train_stats={'step': step,
                                 'acc': report_stats.accuracy(),
                                 'ppl': report_stats.ppl()})

            report_stats = self._maybe_report_training(
                step, train_steps,
                self.optim.learning_rate(),
                report_stats)

            if valid_iter is not None and step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d'
                                % (self.gpu_rank, step))
                valid_stats = self.validate(
                    valid_iter, moving_average=self.moving_average)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat \
                                step %d' % (self.gpu_rank, step))
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: report stat step %d'
                                % (self.gpu_rank, step))
                self._report_step(self.optim.learning_rate(),
                                  step, valid_stats=valid_stats)

                stats_manager.add_stats(
                    valid_stats={'step': step,
                                 'acc': valid_stats.accuracy(),
                                 'ppl': valid_stats.ppl()})
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

            if (self.model_saver is not None
                and (save_checkpoint_steps != 0
                     and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats, stats_manager

    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        if moving_average:
            valid_model = deepcopy(self.model)
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                param.data = avg.data
        else:
            valid_model = self.model

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics(n_latent=self.n_latent)

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                                   else (batch.src, None)
                adjacency = batch.mask[0]
                trues = torch.ones_like(adjacency)
                falses = torch.zeros_like(adjacency)
                final_adj = list()
                if len(self.max_dist) == 0:
                    final_adj = None
                elif len(self.max_dist) == 1:
                    a = torch.where(adjacency < self.max_dist[0] + 1, trues, falses)
                    final_adj = torch.where(adjacency > 0, a, falses)
                else:
                    for dist in self.max_dist:
                        a = torch.where(adjacency == dist + 1, trues, falses)
                        final_adj.append(torch.where(adjacency == 1, trues, a))
                tgt = batch.tgt

                segment_input = None
                if self.segment_token_idx is not None:
                    segment_input = latent_utils.get_segment_input(
                        tgt, self.segment_token_idx)

                if self.n_latent > 1:
                    seq_len, batch_sz, _ = tgt.size()
                    latent_inputs = latent_utils.get_latent_inputs(
                        self.n_latent, seq_len-1, batch_sz)

                    all_losses = []
                    for latent_input in latent_inputs:
                        latent_outputs, _ = valid_model(
                            src, tgt, src_lengths, latent_input=latent_input, segment_input=segment_input,
                            adj=final_adj)
                        latent_losses, _ = self.latent_loss(
                            batch, latent_outputs, None, n_latent=self.n_latent)
                        all_losses.append(latent_losses)
                    all_losses = torch.stack(all_losses, dim=1)
                    max_latent = torch.argmin(all_losses, dim=1)
                    max_latent_input = max_latent.view([1, -1]).repeat([seq_len-1, 1])

                    outputs, attns = valid_model(
                        src, tgt, src_lengths, latent_input=max_latent_input, segment_input=segment_input,
                        adj=final_adj)

                    latent_counts = np.zeros([self.n_latent])
                    for i in range(self.n_latent):
                        latent_counts[i] = torch.sum(max_latent == i).item()
                else:
                    # F-prop through the model.
                    outputs, attns = valid_model(src, tgt, src_lengths, segment_input=segment_input, adj=final_adj)

                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns, n_latent=self.n_latent)

                if self.n_latent > 1:
                    batch_stats.latent_counts += latent_counts

                # Update statistics.
                stats.update(batch_stats)

        if moving_average:
            del valid_model
        else:
            # Set model back to training mode.
            valid_model.train()

        return stats

    def _get_properties(self, src):
        batch_src = src.transpose(0, 1).contiguous()
        adjacency = torch.zeros(batch_src.size(0), batch_src.size(1), batch_src.size(1),
                                device=src.device, dtype=torch.int32)
        for i, sentence in enumerate(batch_src[:, :, 0]):
            sequence = ''
            not_atom_indices = list()
            atom_indices = list()
            pad_indices = list()
            for j, che in enumerate(sentence):
                name = self.inv_stoi[che.item()]
                if che.item() != 1 and che.item() != 0:
                    sequence += self.inv_stoi[che.item()]
                    if (len(name) == 1 and not name.isalpha()) or (len(name) > 1 and name[0] not in ['[', 'B', 'C']):
                        not_atom_indices.append(j)
                    else:
                        atom_indices.append(j)
                elif che.item() == 1 or che.item() == 0:
                    pad_indices.append(j)

            mol = Chem.MolFromSmiles(sequence)
            if mol is None:
                mol = Chem.MolFromSmiles(sequence, sanitize=False)
            adjacency_mol = torch.tensor(GetDistanceMatrix(mol), device=src.device, dtype=torch.int32)
            adjacency_mol += torch.eye(adjacency_mol.shape[0], device=src.device, dtype=torch.int32)
            length = batch_src.size(1)
            for x in range(length):
                for y in range(length):
                    if x in pad_indices or y in pad_indices:
                        adjacency[i, x, y] = 0
                    elif x in atom_indices and y in atom_indices:
                        adjacency[i, x, y] = adjacency_mol[atom_indices.index(x), atom_indices.index(y)]
                    elif x == y and x in not_atom_indices:
                        adjacency[i, :, y] = 1
                        adjacency[i, x, :] = 1
                        adjacency[i, x, y] = 1

        trues = torch.ones_like(adjacency)
        falses = torch.zeros_like(adjacency)
        a = torch.where(adjacency < self.max_dist, trues, falses)
        b = torch.where(adjacency > 0, a, falses)

        return b

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            adjacency = batch.mask[0]
            trues = torch.ones_like(adjacency)
            falses = torch.zeros_like(adjacency)
            final_adj = list()
            if len(self.max_dist) == 0:
                final_adj = None
            elif len(self.max_dist) == 1:
                a = torch.where(adjacency < self.max_dist[0] + 1, trues, falses)
                final_adj = torch.where(adjacency > 0, a, falses)
            else:
                for dist in self.max_dist:
                    a = torch.where(adjacency == dist + 1, trues, falses)
                    final_adj.append(torch.where(adjacency == 1, trues, a))

            # adjacency_check = self._get_properties(src)
            # assert (adjacency == adjacency_check).all()
            tgt_outer = batch.tgt

            bptt = False
            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()

                segment_input = None
                if self.segment_token_idx is not None:
                    segment_input = latent_utils.get_segment_input(
                        tgt, self.segment_token_idx)

                if self.n_latent > 1:
                    self.model.eval()  # Turn off dropout when choosing latent classes
                    seq_len, batch_sz, _ = tgt.size()

                    latent_inputs = latent_utils.get_latent_inputs(
                        self.n_latent, seq_len-1, batch_sz)

                    with torch.no_grad():
                        all_losses = []
                        for latent_input in latent_inputs:
                            latent_outputs, _ = self.model(
                                src, tgt, src_lengths, latent_input=latent_input, segment_input=segment_input,
                                adj=final_adj)
                            latent_losses, _ = self.latent_loss(
                                batch, latent_outputs, None)
                            all_losses.append(latent_losses)
                        all_losses = torch.stack(all_losses, dim=1)
                        max_latent = torch.argmin(all_losses, dim=1)
                        max_latent_input = max_latent.view([1, -1]).repeat([seq_len-1, 1])

                    self.model.train()
                    outputs, attns = self.model(
                        src, tgt, src_lengths, bptt=bptt, latent_input=max_latent_input, segment_input=segment_input,
                        adj=final_adj)

                    latent_counts = np.zeros([self.n_latent])
                    for i in range(self.n_latent):
                        latent_counts[i] = torch.sum(max_latent == i).item()
                else:
                    outputs, attns = self.model(src, tgt, src_lengths, bptt=bptt,
                                                segment_input=segment_input, adj=final_adj)
                bptt = True

                # 3. Compute loss.
                try:
                    loss, batch_stats = self.train_loss(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=j,
                        trunc_size=trunc_size,
                        n_latent=self.n_latent,
                        alpha=self.alpha)
                    if self.n_latent > 1:
                        batch_stats.latent_counts += latent_counts

                    if loss is not None:
                        self.optim.backward(loss)

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d",
                                self.optim.training_step, k)

                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.model.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    self.optim.step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)
