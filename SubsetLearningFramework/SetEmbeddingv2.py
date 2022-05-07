from SubsetLearningFramework.SubsetLearningFramewrok import SubsetLearningFramework, State
import torch
import numpy as np
import networkx as nx


class SetEmbeddingStateV2(State):
    current_set = set({})
    stop_flag = False

    def __init__(self, current_set=None, stop_flag=False):
        if current_set is None:
            current_set = set({})
        self.current_set = {x for x in current_set}  # it makes a copy of set
        self.stop_flag = stop_flag

    def watch(self):
        pass


class SetEmbeddingModelV2(SubsetLearningFramework):
    def __init__(self, node_num, embedding_dim, graph=None, embedding_func=torch.mean,
                 embedding_func_parameters=None, net=None):
        self.node_num = node_num
        self.embedding_dim = embedding_dim
        self.net = net
        if embedding_func is not None:
            self.embedding_func = embedding_func
        else:
            self.embedding_func = self.net.forward

        self.nodes_embedding = torch.tensor(np.random.randn(node_num, embedding_dim), requires_grad=True)
        self.start_embedding = torch.tensor(np.random.randn(embedding_dim), requires_grad=True)
        self.stop_embedding = torch.tensor(np.random.randn(1, embedding_dim), requires_grad=True)

        # from super class
        self.parameters = [self.nodes_embedding, self.start_embedding, self.stop_embedding]
        if embedding_func_parameters is not None:
            for w in embedding_func_parameters:
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

    def action(self, state: SetEmbeddingStateV2, conditional_set=None) -> list:
        current_set = state.current_set
        action_list = []
        current_set_list = list(current_set)

        if 0 == len(current_set):
            # get nbr_list and subset_embedding
            nbr = set(range(self.node_num))
            nbr_list = list(nbr)
            subset_embedding = self.start_embedding
            # we do not add stop embedding
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
            if self.embedding_func == torch.mean or self.embedding_func == torch.sum:
                subset_embedding = self.embedding_func(self.nodes_embedding[current_set_list], dim=0)
            elif self.embedding_func == torch.max or self.embedding_func == torch.min:
                subset_embedding = self.embedding_func(self.nodes_embedding[current_set_list], dim=0).values
            else:
                subset_embedding = torch.mean(self.nodes_embedding[current_set_list], dim=0)
                subset_embedding = self.embedding_func(subset_embedding.to(torch.float32))

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
        return SetEmbeddingStateV2()

    def take_action(self, state: SetEmbeddingStateV2, action):
        new_state = SetEmbeddingStateV2(current_set=state.current_set)
        if action == 'stop':
            new_state.stop_flag = True
        else:
            new_state.current_set.add(action)
        return new_state
