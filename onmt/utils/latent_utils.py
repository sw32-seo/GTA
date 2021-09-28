import torch
import numpy as np

import pdb


def get_segment_input(tgt, segment_token_idx, device='cuda:0'):
    seq_len, batch_sz, _ = tgt.size()
    seq_len -= 1  # tgt

    segment_input = np.zeros([seq_len, batch_sz])
    tgt_compare = tgt.cpu().numpy()

    for batch_idx in range(batch_sz):
        index = 0
        for seq_idx in range(seq_len):
            if tgt_compare[seq_idx, batch_idx] == segment_token_idx:
                index += 1
            if index != 0:
                segment_input[seq_idx, batch_idx] = index
    segment_input = torch.tensor(segment_input, device=device).long()
    return segment_input


def get_latent_inputs(n_latent, seq_len, batch_sz, device='cuda:0'):
    latent_inputs = []
    for idx in range(n_latent):
        latent_input = torch.ones([seq_len, batch_sz], device=device).long() * idx
        latent_inputs.append(latent_input)
    return latent_inputs
