#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat

import os
import codecs
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus, split_mask
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

import pdb


def main(opt):
    ArgumentParser.validate_translate_opts(opt)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    if 'n_latent' not in vars(opt):
        vars(opt)['n_latent'] = vars(opt)['n_translate_latent']
    logger = init_logger(opt.log_file)

    if 'use_segments' not in vars(opt):
        vars(opt)['use_segments'] = opt.n_translate_segments != 0
        vars(opt)['max_segments'] = opt.n_translate_segments

    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
    mask_path = opt.src[:-3] + 'pkl'
    mask_shards = split_mask(mask_path, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards, mask_shards)

    n_latent = opt.n_latent

    if n_latent > 1:
        for latent_idx in range(n_latent):
            output_path = opt.output_dir + '/output_%d' % (latent_idx)
            out_file = codecs.open(output_path, 'w+', 'utf-8')
            translator.out_file = out_file

            for i, (src_shard, tgt_shard, mask_shard) in enumerate(shard_pairs):
                logger.info("Translating shard %d." % i)
                translator.translate(
                    src=src_shard,
                    tgt=tgt_shard,
                    src_dir=opt.src_dir,
                    batch_size=opt.batch_size,
                    attn_debug=opt.attn_debug,
                    latent_idx=latent_idx,
                    mask=mask_shard)
            src_shards = split_corpus(opt.src, opt.shard_size)
            tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
                if opt.tgt is not None else repeat(None)
            mask_shards = split_mask(mask_path, opt.shard_size)
            shard_pairs = zip(src_shards, tgt_shards, mask_shards)
    else:
        output_path = opt.output_dir + '/output'
        out_file = codecs.open(output_path, 'w+', 'utf-8')
        translator.out_file = out_file

        for i, (src_shard, tgt_shard, mask_shard) in enumerate(shard_pairs):
            logger.info("Translating shard %d." % i)
            translator.translate(
                src=src_shard,
                tgt=tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                attn_debug=opt.attn_debug,
                mask=mask_shard)


DEFAULT_TRANSLATE_PARAMS = {
    'batch_size': 64,
    'replace_unk': True,
    'max_length': 200,
    'beam_size': 10,
    'n_best': 10,
    'log_probs': True,
}


def get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    opt = parser.parse_args()
    main(opt)
