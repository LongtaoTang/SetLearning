from abc import abstractmethod, ABCMeta
import networkx as nx
import torch
import random
import numpy as np


class State(object, metaclass=ABCMeta):
    current_set = set({})
    stop_flag = False

    # we will watch the state after we finished a sample, it might help you to debug
    @abstractmethod
    def watch(self):
        pass


class SubsetLearningFramework(object, metaclass=ABCMeta):
    parameters = None
    graph = nx.Graph()
    start_action_list = []
    start_cum_p = np.zeros(len(start_action_list))
    one_node_action_lists = []
    one_node_cum_ps = []

    def train(self, data: list, optimizer=None, batch_size=50, num_of_training=100, num_of_samples=100):
        if optimizer is None:
            optimizer = torch.optim.RMSprop(params=self.parameters)
            print(optimizer.param_groups)
        for training_num in range(num_of_training):
            batch_loss = torch.randn(1) * 0
            batch_count = 0
            update_count = 0
            for subset in data:
                # print(subset)
                samples_log_path = torch.randn(num_of_samples) * 0   # each entity is a log Pr() of one subset
                samples_r = np.ones(num_of_samples)   # each entity is an importance weight of one sample
                for samples_index in range(num_of_samples):
                    state = self.get_start_state()
                    while state.stop_flag is not True:
                        action_list = self.action(state, conditional_set=subset)
                        act_size = len(action_list)
                        # choose one action_tuple form good set of action, where valid is True
                        action_tuple = random.choices([action_list[i] for i in range(act_size)
                                                       if action_list[i][2] is True],
                                                      weights=[action_list[i][1] for i in range(act_size)
                                                               if action_list[i][2] is True],
                                                      k=1)[0]
                        samples_log_path[samples_index] += torch.log(action_tuple[1])
                        # update by time the sum of probability of good set, r should be a real number
                        samples_r[samples_index] *= float(sum([action_list[i][1] for i in range(act_size)
                                                               if action_list[i][2] is True]))
                        state = self.take_action(state, action=action_tuple[0])  # update state
                        # print(str(state.current_set))
                    # state.watch()

                samples_r = samples_r / sum(samples_r)
                # loss = - log Pr()
                batch_loss += - sum([samples_r[i] * samples_log_path[i] for i in range(num_of_samples)])
                batch_count += 1
                if batch_count == batch_size:
                    # we update
                    optimizer.zero_grad()
                    batch_loss = batch_loss / batch_size
                    batch_loss.backward()
                    optimizer.step()
                    print("update: ", update_count)
                    print(self.parameters[1])
                    update_count += 1
                    # reset batch_count and batch_loss
                    batch_count = 0
                    batch_loss = torch.randn(1) * 0

    def predict(self, output_size: int, bias_size=None) -> list:
        self.calculate_pre_info()   # for accelerate
        predict = []
        if bias_size is not None:
            print("we use size bias")
            target_bucket = np.zeros(len(bias_size))
            current_bucket = np.zeros(len(bias_size))
            for i in range(len(bias_size)):
                target_bucket[i] = int(output_size * bias_size[i])
            print("target_bucket: ", target_bucket)

            early_stop = len(bias_size)
            total_count = 0
            while early_stop > 0:
                subset = self.generate(early_stop)
                size = len(subset) - 1
                if current_bucket[size] >= target_bucket[size]:
                    # we reject
                    pass
                else:
                    predict.append(subset)
                    total_count += 1
                    current_bucket[size] += 1
                    if total_count % 10000 == 0:
                        print(total_count)
                        print(current_bucket)
                        print(target_bucket)
                if current_bucket[early_stop - 1] >= target_bucket[early_stop - 1]:
                    early_stop = early_stop - 1
                    print("early_stop: ", early_stop)
        else:
            for i in range(output_size):
                subset = self.generate()
                predict.append(subset)
                if i % 10000 == 0:
                    print(i)
        return predict

    def generate(self, early_stop=None) -> set:
        state = self.for_start()
        while state.stop_flag is not True:
            action_tuple_list = self.action(state)
            act_size = len(action_tuple_list)
            if act_size == 0:
                # we do not have any nbr to jump, Thus we return the current subset
                return state.current_set
            # choose one action
            choose_list = [action_tuple_list[i][0] for i in range(act_size)]
            choose_weight = [action_tuple_list[i][1] for i in range(act_size)]

            # act = random.choices(choose_list, weights=choose_weight, k=1)[0]
            act_index = self.choose_one(choose_weight)
            act = choose_list[act_index]

            # update state
            state = self.take_action(state, act)
            if early_stop is not None and len(state.current_set) > early_stop:
                # restart generate, because the set is too large
                state = self.for_start()
        return state.current_set

    def calculate_pre_info(self):
        start_state = self.get_start_state()
        action_tuple_list = self.action(start_state)
        self.start_action_list = [action_tuple_list[i][0] for i in range(len(action_tuple_list))]
        self.start_cum_p = np.zeros(len(action_tuple_list))
        sum = 0
        for i in range(len(action_tuple_list)):
            sum += float(action_tuple_list[i][1])
            self.start_cum_p[i] = sum

        # self.one_node_action_lists = []
        # self.one_node_cum_ps = []
        # for i in range(len(action_tuple_list)):
        #     one_node_state = self.take_action(state=start_state, action=self.start_action_list[i])
        #     one_node_action_tuple_list = self.action(state=one_node_state)
        #     self.one_node_action_lists.append([one_node_action_tuple_list[i][0]
        #                                       for i in range(len(one_node_action_tuple_list))])
        #     self.one_node_cum_ps.append(np.zeros(len(one_node_action_tuple_list)))
        #     sum = 0
        #     for j in range(len(one_node_action_tuple_list)):
        #         sum += float(one_node_action_tuple_list[j][1])
        #         self.one_node_cum_ps[i][j] = sum

    def for_start(self):
        state = self.get_start_state()
        act_index = random.choices(range(len(self.start_action_list)), cum_weights=self.start_cum_p, k=1)[0]
        state = self.take_action(state=state, action=self.start_action_list[act_index])
        # act = random.choices(self.one_node_action_lists[act_index], cum_weights=self.one_node_cum_ps[act_index])[0]
        # state = self.take_action(state=state, action=act)
        return state

    def choose_one(self, weights):
        N = len(weights)
        random_num = random.random()
        sum = 0
        for i in range(N):
            sum += weights[i]
            if sum > random_num:
                return i
        return 0

    # it should return: a list like [(act0, p0, valid0), (act1, p1, valid1), ... ]
    # act0:
    # if you are choosing a node, return an integer
    # if you want to stop, return 'stop'
    # else, return None
    # p0:
    # the probability of take this action, it should be a torch tensor type
    # valid0：
    # if conditional_set is not None:
    # it is Ture when this state is in good set, which means it has chance to reach conditional_set
    # if conditional_set is None:
    # always Ture
    @abstractmethod
    def action(self, state, conditional_set=None) -> list:
        pass

    # you should return the start state, which is a State class object
    @abstractmethod
    def get_start_state(self):
        pass

    # update the state by taking the action
    @abstractmethod
    def take_action(self, state, action):
        pass
