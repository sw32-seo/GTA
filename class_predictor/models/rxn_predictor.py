import torch
import torch.nn as nn

import class_predictor.graph.mol_features as mol_features

import pdb


class RxnPredictor(nn.Module):
    def __init__(self, args, n_classes=1):
        super(RxnPredictor, self).__init__()
        self.args = args

        self.src_graph_conv = GraphConv(args)
        if args.share_embed:
            self.tgt_graph_conv = self.src_graph_conv
        else:
            self.tgt_graph_conv = GraphConv(args)

        hidden_size = args.hidden_size
        self.r_h = nn.Linear(hidden_size, hidden_size)
        self.r_o = nn.Linear(hidden_size, n_classes)

    def aggregate_atom_h(self, atom_h, scope):
        mol_h = []
        for (st, le) in scope:
            cur_atom_h = atom_h.narrow(0, st, le)
            mol_h.append(cur_atom_h.sum(dim=0))
        mol_h = torch.stack(mol_h, dim=0)
        return mol_h

    def get_graph_embeds(self, graphs, conv_model):
        graph_inputs, scope = graphs.get_graph_inputs()

        atom_h = conv_model(graph_inputs)
        mol_h = self.aggregate_atom_h(atom_h, scope)
        return mol_h

    def forward(self, src_graphs, tgt_graphs, args):
        src_mol_h = self.get_graph_embeds(src_graphs, self.src_graph_conv)
        tgt_mol_h = self.get_graph_embeds(tgt_graphs, self.tgt_graph_conv)

        rxn_h = tgt_mol_h - src_mol_h
        rxn_h = nn.ReLU()(self.r_h(rxn_h))
        rxn_o = self.r_o(rxn_h)
        return rxn_o


class GraphConv(nn.Module):
    def __init__(self, args):
        """Creates graph conv layers for molecular graphs."""
        super(GraphConv, self).__init__()
        self.args = args
        hidden_size = args.hidden_size

        self.n_atom_feats = mol_features.N_ATOM_FEATS
        self.n_bond_feats = mol_features.N_BOND_FEATS

        # Weights for the message passing network
        self.W_message_i = nn.Linear(self.n_atom_feats + self.n_bond_feats,
                                     hidden_size, bias=False,)
        self.W_message_h = nn.Linear(hidden_size, hidden_size, bias=False,)
        self.W_message_o = nn.Linear(self.n_atom_feats + hidden_size,
                                     hidden_size)

        self.dropout = nn.Dropout(args.dropout)

    def index_select_nei(self, input, dim, index):
        # Reshape index because index_select expects a 1-D tensor. Reshape the
        # output afterwards.
        target = torch.index_select(
            input=input,
            dim=0,
            index=index.view(-1)
        )
        return target.view(index.size() + input.size()[1:])

    def forward(self, graph_inputs):
        fatoms, fbonds, agraph, bgraph = graph_inputs

        # nei_input_h is size [# bonds, hidden_size]
        nei_input_h = self.W_message_i(fbonds)
        # message_h is size [# bonds, hidden_size]
        message_h = nn.ReLU()(nei_input_h)

        for i in range(self.args.depth - 1):
            # nei_message_h is [# bonds, # max neighbors, hidden_size]
            nei_message_h = self.index_select_nei(
                input=message_h,
                dim=0,
                index=bgraph)

            # Sum over the nieghbors, now [# bonds, hidden_size]
            nei_message_h = nei_message_h.sum(dim=1)
            nei_message_h = self.W_message_h(nei_message_h)  # Shared weights

            message_h = nn.ReLU()(nei_input_h + nei_message_h)

        # Collect the neighbor messages for atom aggregation
        nei_message_h = self.index_select_nei(
            input=message_h,
            dim=0,
            index=agraph,
        )
        # Aggregate the messages
        nei_message_h = nei_message_h.sum(dim=1)
        atom_input = torch.cat([fatoms, nei_message_h], dim=1)
        atom_input = self.dropout(atom_input)

        atom_h = nn.ReLU()(self.W_message_o(atom_input))
        return atom_h
