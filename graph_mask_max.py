import pickle
import os
import argparse
import torch

from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.Chem.rdFMCS import FindMCS
from rdkit.Chem.rdmolops import GetDistanceMatrix


def get_adjmask(sequences):
    chars = sequences.strip().split(' ')
    if sequences[0] == '<':
        sequence = ''.join(chars[1:])
    else:
        sequence = ''.join(chars)
    adjacency = torch.zeros(len(chars), len(chars))
    not_atom_indices = list()
    atom_indices = list()
    pad_indices = list()
    for j, cha in enumerate(chars):
        if (len(cha) == 1 and not cha.isalpha()) or (len(cha) > 1 and cha[0] not in ['[', 'B', 'C']):
            not_atom_indices.append(j)
        else:
            atom_indices.append(j)

    mol = Chem.MolFromSmiles(sequence)
    if mol is None:
        mol = Chem.MolFromSmiles(sequence, sanitize=False)

    # distance of Atom tokens start from 2. distance 1 is equal to 2 in adjacency_mol.
    adjacency_mol = torch.tensor(GetDistanceMatrix(mol)) + 1
    adjacency_mol += torch.eye(adjacency_mol.shape[0])
    length = len(chars)
    for x in range(length):
        for y in range(length):
            if x in pad_indices or y in pad_indices:
                adjacency[x, y] = 0
            elif x in atom_indices and y in atom_indices:
                adjacency[x, y] = adjacency_mol[atom_indices.index(x), atom_indices.index(y)]
            elif x == y and x in not_atom_indices:
                adjacency[:, y] = 1
                adjacency[x, :] = 1
                adjacency[x, y] = 1

    return adjacency


def get_atom_map(src, tgt):
    src_chars = src.strip().split(' ')
    tgt_chars = tgt.strip().split(' ')
    if src[0] == '<':
        src_smi = ''.join(src_chars[1:])
    else:
        src_smi = ''.join(src_chars)
    tgt_smi = ''.join(tgt_chars)
    tgt_mols = Chem.MolFromSmiles(tgt_smi)
    tgt_smis = tgt_smi.split('.')
    src_mol = Chem.MolFromSmiles(src_smi)
    atom_map = torch.zeros(src_mol.GetNumAtoms(), tgt_mols.GetNumAtoms())
    cross_attn = torch.zeros(len(src_chars), len(tgt_chars))
    not_atom_indices_src = list()
    atom_indices_src = list()
    pad_indices_src = list()
    not_atom_indices_tgt = list()
    atom_indices_tgt = list()
    pad_indices_tgt = list()
    for smi in tgt_smis:
        tgt_mol = Chem.MolFromSmiles(smi)
        mols = [src_mol, tgt_mol]
        result = FindMCS(mols, timeout=10)
        result_mol = Chem.MolFromSmarts(result.smartsString)
        src_mat = src_mol.GetSubstructMatches(result_mol)
        tgt_mat = tgt_mols.GetSubstructMatches(result_mol)
        if len(src_mat) > 0 and len(tgt_mat) > 0:
            for i, j in zip(src_mat[0], tgt_mat[0]):
                atom_map[i, j] = 1
    # match = atom_map.sum(0)
    # for i in range(match.size(0)):
    #     if match[i] == 0:
    #         atom_map[:, i] = 1

    for j, cha in enumerate(src_chars):
        if (len(cha) == 1 and not cha.isalpha()) or (len(cha) > 1 and cha[0] not in ['[', 'B', 'C']):
            not_atom_indices_src.append(j)
        else:
            atom_indices_src.append(j)
    for j, cha in enumerate(tgt_chars):
        if (len(cha) == 1 and not cha.isalpha()) or (len(cha) > 1 and cha[0] not in ['[', 'B', 'C']):
            not_atom_indices_tgt.append(j)
        else:
            atom_indices_tgt.append(j)
    for x in range(len(src_chars)):
        for y in range(len(tgt_chars)):
            if x in pad_indices_src or y in pad_indices_tgt:
                cross_attn[x, y] = 0
            elif x in atom_indices_src and y in atom_indices_tgt:
                cross_attn[x, y] = atom_map[atom_indices_src.index(x), atom_indices_tgt.index(y)]
            elif x in not_atom_indices_src and y in not_atom_indices_tgt:
                cross_attn[:, y] = 0
                cross_attn[x, :] = 0
                cross_attn[x, y] = 0

    return cross_attn


parser = argparse.ArgumentParser(description="Get database dir")
parser.add_argument('--data', '-data', type=str, default='data/aug_shift-x2P2R_no_stereo')
parser.add_argument('--test', '-test', action='store_true', default=False)
args = parser.parse_args()

database = args.data
is_test = args.test

if is_test:
    path = os.path.join(database, 'tgt-test.txt')
    with open(path, "r") as f:
        tgt_lines = f.readlines()

    path = os.path.join(database, 'src-test.txt')
    with open(path, "r") as f:
        src_lines = f.readlines()

    with open(path[:-3] + 'pkl', 'wb') as f:
        masks = list()
        for line in tqdm(src_lines):
            mask = get_adjmask(line)
            masks.append(mask)
        pickle.dump(masks, f)

    del masks

else:

    path = os.path.join(database, 'tgt-train.txt')
    with open(path, "r") as f:
        tgt_lines = f.readlines()

    path = os.path.join(database, 'src-train.txt')
    with open(path, "r") as f:
        src_lines = f.readlines()

    with open(path[:-3]+'pkl_cross', 'wb') as f:
        masks = list()
        for src_line, tgt_line in zip(tqdm(src_lines), tgt_lines):
            mask = get_atom_map(src_line, tgt_line)
            masks.append(mask)
        pickle.dump(masks, f)

    del masks

    with open(path[:-3]+'pkl', 'wb') as f:
        masks = list()
        for line in tqdm(src_lines):
            mask = get_adjmask(line)
            masks.append(mask)
        pickle.dump(masks, f)

    del masks

    path = os.path.join(database, 'tgt-val.txt')
    with open(path, "r") as f:
        tgt_lines = f.readlines()

    path = os.path.join(database, 'src-val.txt')
    with open(path, "r") as f:
        src_lines = f.readlines()

    with open(path[:-3]+'pkl', 'wb') as f:
        masks = list()
        for line in tqdm(src_lines):
            mask = get_adjmask(line)
            masks.append(mask)
        pickle.dump(masks, f)

    del masks

    with open(path[:-3]+'pkl_cross', 'wb') as f:
        masks = list()
        for src_line, tgt_line in zip(tqdm(src_lines), tgt_lines):
            mask = get_atom_map(src_line, tgt_line)
            masks.append(mask)
        pickle.dump(masks, f)

    del masks

    path = os.path.join(database, 'tgt-test.txt')
    with open(path, "r") as f:
        tgt_lines = f.readlines()

    path = os.path.join(database, 'src-test.txt')
    with open(path, "r") as f:
        src_lines = f.readlines()

    with open(path[:-3]+'pkl', 'wb') as f:
        masks = list()
        for line in tqdm(src_lines):
            mask = get_adjmask(line)
            masks.append(mask)
        pickle.dump(masks, f)

    del masks
