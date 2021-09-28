import re
import argparse
import os
from tqdm import tqdm

from rdkit import Chem
from rdkit import rdBase


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)
    
    
def random_smiles(smi_):
    mol_ = Chem.MolFromSmiles(smi_.replace(" ", "").replace("\n", ""))
    
    flag = 0
    while flag < 10:
        smi__ = Chem.MolToSmiles(mol_, doRandom=True)
        flag += 1
        if smi__ != smi_:
            flag = 10
    
    return smi_tokenizer(smi__)

    
parser = argparse.ArgumentParser(description="Get database dir")
parser.add_argument('--ori_data', '-ori_data', type=str, default='data/aug_shift-x2P2R')
parser.add_argument('--mode', '-mode', type=str, default='non-stereo',
                    choices=['shift', '2p', '2r', '2p2r', '2p2r_shift', 'non-stereo'])
args = parser.parse_args()

filepath = args.ori_data

# jaa case: reactants = tgt, product = src
srcfn = ['src-train.txt']
tgtfn = ['tgt-train.txt']

if args.mode == 'non-stereo':

    srcfn = ['src-train.txt', 'src-val.txt', 'src-test.txt']
    tgtfn = ['tgt-train.txt', 'tgt-val.txt', 'tgt-test.txt']

    savepath = args.ori_data + '_no_stereo'

    if not os.path.exists(savepath):
        os.mkdir(savepath)

    for i in range(len(tgtfn)):
        ft = open(os.path.join(filepath, tgtfn[i]), 'r')
        tlines = ft.readlines()
        ft.close()
        fs = open(os.path.join(filepath, srcfn[i]), 'r')
        slines = fs.readlines()
        fs.close()

        for sline, tline in zip(tqdm(slines), tlines):
            src = ''.join(sline.strip().split(' '))
            tgt = ''.join(tline.strip().split(' '))

            src_smiles = src.split('.')
            tgt_smiles = tgt.split('.')

            new_src_smiles = ''
            for src_smile in src_smiles:
                mol = Chem.MolFromSmiles(src_smile)
                if len(new_src_smiles) > 0:
                    new_src_smiles = '.'.join((new_src_smiles, Chem.MolToSmiles(mol, isomericSmiles=False)))
                else:
                    new_src_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

            new_tgt_smiles = ''
            for tgt_smile in tgt_smiles:
                mol = Chem.MolFromSmiles(tgt_smile)
                if len(new_tgt_smiles) > 0:
                    new_tgt_smiles = '.'.join((new_tgt_smiles, Chem.MolToSmiles(mol, isomericSmiles=False)))
                else:
                    new_tgt_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

            with open(os.path.join(savepath, tgtfn[i]), 'a') as new_fs:
                new_fs.write(smi_tokenizer(new_tgt_smiles) + '\n')

            with open(os.path.join(savepath, srcfn[i]), 'a') as new_ft:
                new_ft.write(smi_tokenizer(new_src_smiles) + '\n')


if args.mode == 'shift':
    # ori / aug_shift-
    # p -> r1.r2 or r2.r1 and r1 and r1.r2.r3 combinations too

    savepath = args.ori_data + '_shift'

    if not os.path.exists(savepath):
        os.mkdir(savepath)

    printflag = 0
    pflag = 0

    for i in range(len(tgtfn)):
        fs = open(os.path.join(filepath, tgtfn[i]), 'r')
        slines = fs.readlines()
        ft = open(os.path.join(filepath, srcfn[i]), 'r')
        flines = ft.readlines()

        for i_, sline in enumerate(slines):
            src = sline.split('.')

            if len(src) == 2:
                src[0] = src[0][:-1]
                src[1] = src[1][1:-1]

                with open(os.path.join(savepath, tgtfn[i]), 'a') as new_fs:
                    new_fs.write(" . ".join((src[0], src[1])) + '\n')
                    new_fs.write(" . ".join((src[1], src[0])) + '\n')

                with open(os.path.join(savepath, srcfn[i]), 'a') as new_ft:
                    new_ft.write(flines[i_][:-1] + '\n')
                    new_ft.write(flines[i_][:-1] + '\n')

            elif len(src) == 1:
                src[0] = src[0][:-1]

                with open(os.path.join(savepath, tgtfn[i]), 'a') as new_fs:
                    new_fs.write(src[0] + '\n')
                with open(os.path.join(savepath, srcfn[i]), 'a') as new_ft:
                    new_ft.write(flines[i_][:-1] + '\n')

            elif len(src) == 3:
                src[0] = src[0][:-1]
                src[1] = src[1][1:-1]
                src[2] = src[2][1:-1]

                with open(os.path.join(savepath, tgtfn[i]), 'a') as new_fs:
                    new_fs.write(" . ".join((src[0], src[1], src[2])) + '\n')
                    new_fs.write(" . ".join((src[0], src[2], src[1])) + '\n')
                    new_fs.write(" . ".join((src[1], src[0], src[2])) + '\n')
                    new_fs.write(" . ".join((src[1], src[2], src[0])) + '\n')
                    new_fs.write(" . ".join((src[2], src[0], src[1])) + '\n')
                    new_fs.write(" . ".join((src[2], src[1], src[0])) + '\n')

                with open(os.path.join(savepath, srcfn[i]), 'a') as new_ft:
                    new_ft.write(flines[i_][:-1] + '\n')
                    new_ft.write(flines[i_][:-1] + '\n')
                    new_ft.write(flines[i_][:-1] + '\n')
                    new_ft.write(flines[i_][:-1] + '\n')
                    new_ft.write(flines[i_][:-1] + '\n')
                    new_ft.write(flines[i_][:-1] + '\n')

        fs.close()
        ft.close()

elif args.mode == '2p2r_shift':
    # x(n)P(n)R + aug_shift-
    # p or p' -> r1.r2/r2.r1 or r1'.r2'/r2'.r1' , r1 or r1'

    for n in [2]:
        savepath = args.ori_data + '_' + str(n) + 'P' + str(n) + 'R_shift/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        for i in range(len(srcfn)):
            fs = open(os.path.join(filepath, tgtfn[i]), 'r')
            slines = fs.readlines()
            ft = open(os.path.join(filepath, srcfn[i]), 'r')
            tlines = ft.readlines()

            for i_, sline in enumerate(slines):
                src = sline.split('.')

                if len(src) == 2:

                    src[0] = src[0][:-1]
                    src[1] = src[1][1:-1]

                    with open(os.path.join(savepath, tgtfn[i]), 'a') as new_fs:
                        new_fs.write(" . ".join((src[0], src[1])) + '\n')
                        new_fs.write(" . ".join((src[1], src[0])) + '\n')

                    with open(os.path.join(savepath, srcfn[i]), 'a') as new_ft:
                        new_ft.write(tlines[i_][:-1] + '\n')
                        new_ft.write(tlines[i_][:-1] + '\n')

                    for n_ in range(n - 1):
                        r_src_0 = random_smiles(src[0])
                        r_src_1 = random_smiles(src[1])
                        if tlines[i_][0] == '<':
                            rxn_cls = tlines[i_].split(' ')[0]
                            new_line = ' '.join(tlines[i_][:-1].split(' ')[1:])
                            r_tgt = random_smiles(new_line)
                        else:
                            r_tgt = random_smiles(tlines[i_][:-1])
                            rxn_cls = None

                        with open(os.path.join(savepath, tgtfn[i]), 'a') as new_fs:
                            new_fs.write(" . ".join((r_src_0, r_src_1)) + '\n')
                            new_fs.write(" . ".join((r_src_1, r_src_0)) + '\n')

                        with open(os.path.join(savepath, srcfn[i]), 'a') as new_ft:
                            if rxn_cls is not None:
                                new_ft.write(rxn_cls + ' ' + r_tgt + '\n')
                                new_ft.write(rxn_cls + ' ' + r_tgt + '\n')
                            else:
                                new_ft.write(r_tgt + '\n')
                                new_ft.write(r_tgt + '\n')

                elif len(src) == 1:
                    src[0] = src[0][:-1]

                    with open(os.path.join(savepath, tgtfn[i]), 'a') as new_fs:
                        new_fs.write(src[0] + '\n')

                    with open(os.path.join(savepath, srcfn[i]), 'a') as new_ft:
                        new_ft.write(tlines[i_][:-1] + '\n')

                    for n_ in range(n - 1):
                        r_src_0 = random_smiles(src[0])
                        if tlines[i_][0] == '<':
                            rxn_cls = tlines[i_].split(' ')[0]
                            new_line = ' '.join(tlines[i_][:-1].split(' ')[1:])
                            r_tgt = random_smiles(new_line)
                        else:
                            r_tgt = random_smiles(tlines[i_][:-1])
                            rxn_cls = None

                        with open(os.path.join(savepath, tgtfn[i]), 'a') as new_fs:
                            new_fs.write(r_src_0 + '\n')

                        with open(os.path.join(savepath, srcfn[i]), 'a') as new_ft:
                            if rxn_cls is not None:
                                new_ft.write(rxn_cls + ' ' + r_tgt + '\n')
                            else:
                                new_ft.write(r_tgt + '\n')

                else:
                    src[0] = src[0][:-1]
                    src[1] = src[1][1:-1]
                    src[2] = src[2][1:-1]

                    with open(os.path.join(savepath, tgtfn[i]), 'a') as new_fs:
                        new_fs.write(" . ".join((src[0], src[1], src[2])) + '\n')
                        new_fs.write(" . ".join((src[0], src[2], src[1])) + '\n')
                        new_fs.write(" . ".join((src[1], src[0], src[2])) + '\n')
                        new_fs.write(" . ".join((src[1], src[2], src[0])) + '\n')
                        new_fs.write(" . ".join((src[2], src[0], src[1])) + '\n')
                        new_fs.write(" . ".join((src[2], src[1], src[0])) + '\n')

                    with open(os.path.join(savepath, srcfn[i]), 'a') as new_ft:
                        new_ft.write(tlines[i_][:-1] + '\n')
                        new_ft.write(tlines[i_][:-1] + '\n')
                        new_ft.write(tlines[i_][:-1] + '\n')
                        new_ft.write(tlines[i_][:-1] + '\n')
                        new_ft.write(tlines[i_][:-1] + '\n')
                        new_ft.write(tlines[i_][:-1] + '\n')

                    for n_ in range(n - 1):
                        r_src_0 = random_smiles(src[0])
                        r_src_1 = random_smiles(src[1])
                        r_src_2 = random_smiles(src[2])
                        if tlines[i_][0] == '<':
                            rxn_cls = tlines[i_].split(' ')[0]
                            new_line = ' '.join(tlines[i_][:-1].split(' ')[1:])
                            r_tgt = random_smiles(new_line)
                        else:
                            r_tgt = random_smiles(tlines[i_][:-1])
                            rxn_cls = None

                        with open(os.path.join(savepath, tgtfn[i]), 'a') as new_fs:
                            new_fs.write(" . ".join((r_src_0, r_src_1, r_src_2)) + '\n')
                            new_fs.write(" . ".join((r_src_0, r_src_2, r_src_1)) + '\n')
                            new_fs.write(" . ".join((r_src_1, r_src_0, r_src_2)) + '\n')
                            new_fs.write(" . ".join((r_src_1, r_src_2, r_src_0)) + '\n')
                            new_fs.write(" . ".join((r_src_2, r_src_0, r_src_1)) + '\n')
                            new_fs.write(" . ".join((r_src_2, r_src_1, r_src_0)) + '\n')

                        with open(os.path.join(savepath, srcfn[i]), 'a') as new_ft:
                            if rxn_cls is not None:
                                new_ft.write(rxn_cls + ' ' + r_tgt + '\n')
                                new_ft.write(rxn_cls + ' ' + r_tgt + '\n')
                                new_ft.write(rxn_cls + ' ' + r_tgt + '\n')
                                new_ft.write(rxn_cls + ' ' + r_tgt + '\n')
                                new_ft.write(rxn_cls + ' ' + r_tgt + '\n')
                                new_ft.write(rxn_cls + ' ' + r_tgt + '\n')
                            else:
                                new_ft.write(r_tgt + '\n')
                                new_ft.write(r_tgt + '\n')
                                new_ft.write(r_tgt + '\n')
                                new_ft.write(r_tgt + '\n')
                                new_ft.write(r_tgt + '\n')
                                new_ft.write(r_tgt + '\n')

            fs.close()
            ft.close()
