from SubsetLearningFramework.SubsetLearningFramewrok import SubsetLearningFramework, State
import torch
import numpy as np
import networkx as nx
import torch.nn as nn


# -----------------------------------------------------------------------------------------------------
# net
class GRUNet(nn.Module):
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        super(GRUNet, self).__init__()
        self.R_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.Z_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.H_tilde_gate = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, x, H):
        R = torch.sigmoid(self.R_gate(torch.concat((x, H))))
        Z = torch.sigmoid(self.Z_gate(torch.concat((x, H))))
        H_tilde = torch.tanh(self.H_tilde_gate(torch.concat((x, R * H))))
        return Z * H + (1-Z) * H_tilde


class GRUStateV2(State):
    current_set = set({})
    stop_flag = False
    H = None

    def __init__(self, current_set=None, stop_flag=False, H=None):
        if current_set is None:
            current_set = set({})
        self.current_set = {x for x in current_set}     # it makes a copy of set
        self.stop_flag = stop_flag
        self.H = H

    def watch(self):
        pass


class GRUmodelV2(SubsetLearningFramework):
    def __init__(self, node_num, embedding_dim, graph=None):
        self.node_num = node_num
        self.embedding_dim = embedding_dim

        self.nodes_embedding = torch.tensor(np.random.randn(node_num, embedding_dim), requires_grad=True)
        self.start_H = torch.tensor(np.random.randn(embedding_dim), requires_grad=True)
        self.stop_embedding = torch.tensor(np.random.randn(1, embedding_dim), requires_grad=True)

        # for the net
        self.net = GRUNet(embedding_dim)

        # from super class
        self.parameters = [self.nodes_embedding, self.start_H, self.stop_embedding]
        for w in self.net.parameters():
            self.parameters.append(w)

        if graph is None:
            # we build a fully connected graph
            self.graph = nx.Graph()
            self.graph.add_node(node_num)
            for x in range(node_num):
                for y in range(node_num):
                    if x < y:
                        self.graph.add_edge(x, y)
        else:
            self.graph = graph

    def action(self, state: GRUStateV2, conditional_set=None) -> list:
        current_set = state.current_set
        action_list = []
        current_set_list = list(current_set)
        if 0 == len(current_set):
            # get nbr_list and subset_embedding
            nbr = set(range(self.node_num))
            nbr_list = list(nbr)
            subset_embedding = self.start_H
            # calculate p
            p = self.nodes_embedding[nbr_list] * subset_embedding
            p = torch.sum(p, dim=1)
            p = torch.softmax(p, dim=-1)
            for i in range(len(nbr_list)):
                if conditional_set is not None:
                    action_list.append((nbr_list[i], p[i], nbr_list[i] in conditional_set))
                else:
                    action_list.append((nbr_list[i], p[i], True))
        else:
            # get all neighbors
            nbr = set({})
            for x in current_set:
                nbr = nbr | set(self.graph.neighbors(x))
            nbr = nbr - current_set
            nbr_list = list(nbr)
            # get current set embedding
            subset_embedding = state.H
            # we need to add stop embedding
            # calculate p
            p = torch.concat((self.nodes_embedding[nbr_list], self.stop_embedding)) * subset_embedding
            p = torch.sum(p, dim=1)
            p = torch.softmax(p, dim=-1)

            if conditional_set is not None:
                if len(conditional_set) == len(current_set):
                    # we can stop
                    action_list.append(('stop', p[-1], True))
                else:
                    # we can not stop
                    action_list.append(('stop', p[-1], False))
                for i in range(len(nbr_list)):
                    action_list.append((nbr_list[i], p[i], nbr_list[i] in conditional_set))
            else:
                action_list.append(('stop', p[-1], True))
                for i in range(len(nbr_list)):
                    action_list.append((nbr_list[i], p[i], True))
        return action_list

    def get_start_state(self):
        return GRUStateV2(H=self.start_H)

    def take_action(self, state:GRUStateV2, action):
        new_state = GRUStateV2(current_set=state.current_set, H=state.H)
        if action == 'stop':
            new_state.stop_flag = True
        else:
            new_state.current_set.add(action)
            new_state.H = self.net.forward(x=self.nodes_embedding[action].to(torch.float32),
                                           H=state.H.to(torch.float32))
        return new_state
