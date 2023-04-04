from typing import OrderedDict
import pandas as pd
import arg_explainer as ae
import networkx as nx
from sklearn import preprocessing

from itertools import combinations
import numpy as np
import utils

import dataset_manager


class Testbench(object):
    """
    Tool functions to test the arg_explainer class and data-based setups.
    """

    def __init__(self, data_manager, exp_name=None) -> None:
        self.dm = data_manager
        self.classifier = self.dm.get_classifier()
        self.exp_name = exp_name
        pass

    def explore_full_dataset(self, nb_steps):
        """
        nb_rows: in initial dataset
        """
        full_dataset = self.dm.generate_full_dataset()
        full_dataset_encoded = [self.dm.encode(i) for i in full_dataset]

        # Shuffle the dataset
        full_dataset_shuff = utils.shuffle_dataset(full_dataset_encoded, seed=1)

        for nb_rows in utils.make_slices(self.dm.space_size(), nb_steps):

            self.dm.use_synth_dataset(full_dataset_shuff, nb_rows)
            explainer = ae.ArgTabularExplainer(self.dm, self.exp_name + '_' + str(nb_rows) + '_fullsynth', compute=True, output_path='../../saves')

            G = explainer.build_attack_graph(compute=True, display_graph=False)
            print('total args:', len(G.nodes()))
            print('edges per node:', np.mean([len(G.edges(n)) for n in G.nodes()]))


    def generate_full_dataset(self):
        # TODO: make yield version
        instance = OrderedDict.fromkeys(self.feature_names)
        full_dataset = []
        def gen_rec(instance, features, f_i, full_dataset):
            inst_ = instance.copy()
            if f_i == len(features) - 1:
                for f in self.fvalues_p_col[features[f_i]]:
                    inst_[features[f_i]] = self.feature_value_names[f][len(features[f_i])+1:]
                    full_dataset.append(list(inst_.values()))
            else:
                inst_ = instance.copy()
                for f in self.fvalues_p_col[features[f_i]]:
                    inst_[features[f_i]] = self.feature_value_names[f][len(features[f_i])+1:]
                    gen_rec(inst_, features, f_i + 1, full_dataset)

        gen_rec(instance, list(instance.keys()), 0, full_dataset)
        return full_dataset
    

    def explore_neighborhoods(self, nb_steps):

        origin_dataset = self.dm.dataset
        nbh_dataset = []

        depth = -1
        for nb_rows in utils.make_slices(self.dm.space_size(), nb_steps):
            #generate nbh dataset with enough data
            while len(nbh_dataset) < nb_rows:
                depth += 1
                nbh_dataset = self.dm.generate_neighborhoods(origin_dataset, depth)
                print(len(nbh_dataset), nb_rows)
                #nbh_dataset = utils.remove_duplicates(nbh_dataset)
        
            # Shuffle the dataset
            nbh_dataset_shuff = utils.shuffle_dataset(nbh_dataset, seed=1) 
            
            print('nbh dataset :', nbh_dataset_shuff[0])
            self.dm.use_synth_dataset(nbh_dataset_shuff, nb_rows)
            explainer = ae.ArgTabularExplainer(self.dm, self.exp_name + '_' + str(nb_rows) + '_synthnbh', compute=True, output_path='../../saves')

            G = explainer.build_attack_graph(compute=True, display_graph=False)
            print('total args:', len(G.nodes()))
            print('edges per node:', np.mean([len(G.edges(n)) for n in G.nodes()]))
            
    
    def generate_neighborhoods(self, dataset, depth):
        nbh_dataset = []
        print('Generating ' + str(len(dataset)) + ' neighborhoods...')
        for inst_ in dataset:
            origin = self.data2origin(inst_)
            nbh_dataset += self.dm.generate_full_neighborhood(origin, depth)
            # try:
            #     origin = self.data2origin(inst_)
            #     nbh_dataset += self.dm.generate_full_neighborhood(origin, depth)
            # except:
            #     print('error', inst_)
        return nbh_dataset
    

    def data2origin(self, instance):
        origin = OrderedDict.fromkeys(self.dm.feature_names)
        for k, v in zip(self.dm.feature_names, instance):
            fk = list(self.dm.values_p_feature[k])[int(v)]
            #origin[k] = self.dm.feature_names[fk][len(str(k))+1:]
            origin[k] = self.dm.feature_value_names[fk][len(str(k))+1:]
        return origin


    # def generate_full_neighborhood(origin, depth):
    #     instance = origin.copy()
    #     nbh_list = []

    #     def gen_rec(self, cols_, nbh_list):
    #         nbhs = []
    #         for nbh in nbh_list:
    #             for f in self.fvalues_p_col[cols_[0]]:
    #                 # create a new instance with each value for the column
    #                 current_value = self.feature_value_names[f][len(cols_[0])+1:]
    #                 if current_value != origin[cols_[0]]:
    #                     inst_ = nbh.copy()
    #                     inst_[cols_[0]] = current_value
    #                     nbhs.append(inst_)
    #         if len(cols_) > 1:
    #             nbhs = gen_rec(cols_[1:], nbhs)
    #         return nbhs
                    
    #     for cols_ in combinations(list(instance.keys()), depth):
    #         nbh_list += gen_rec(cols_, [instance])
    #     nlist = []
    #     for nbh in nbh_list:
    #         nlist.append(list(nbh.values()))
    #     return nlist
    



    def generate_instance_random(self, constraints):
        instance = OrderedDict.fromkeys(self.explainer.dataset.columns)
        for f in constraints:
            col=self.explainer.col_p_feature[f]
            instance[col] = self.explainer.feature_names[f].split('_')[1]
            self.fill_instance(instance, 'random')
        return list(instance.values())

    def fill_instance(self, instance, strategy):
        if strategy == 'random':
            # fill the instance with a random values sampled from baseline dataset
            for i_, col in enumerate(instance.keys()):
                if instance[col] is None:
                    instance[col] = self.dataset[col].sample(1, random_state=1).values[0]
        elif strategy == 'most_frequent':
            # fill the instance with the most frequent value for each column
            for k, i_ in enumerate(instance.keys()):
                if instance[k] is None:
                    instance[k] = self.dataset[i_].value_counts().index[0]
        
    
    

    # def explore_full_dataset(self, nb_steps):
    #     full_dataset = self.generate_full_dataset()
    #     y_plus = [c.predict(self.instance2encoded(i_, self.dataset).reshape(1,-1))[0] for i_ in full_dataset]
    #     #train_data_plus = train_data = pd.DataFrame(transformed_data[:nb_vals] + transformed_data_plus, columns=dataset.feature_names)
        
    #     random.seed(1)
    #     indices = list(range(len(full_dataset)))
    #     random.shuffle(indices)
    #     full_dataset_shuff = [full_dataset[i] for i in indices] 
    #     y_plus_shuff = [y_plus[i] for i in indices]

    #     print(np.unique(np.array(y_plus_shuff)))
        
    #     step_len = len(full_dataset)//nb_steps
    #     steps = [i*step_len for i in range(1, nb_steps)]
    #     steps.append(len(full_dataset))
    #     for nb_vals in steps:
    #         dataset_t = full_dataset_shuff[:nb_vals]
    #         y_t = y_plus_shuff[:nb_vals]
    #         print('total length', len(dataset_t), len(y_t))
    #         train_data_plus = pd.DataFrame(dataset_t, columns=self.dataset.feature_names)

    #         explainer = ae.ArgTabularExplainer(self.classifier, train_data_plus, y_t, 'titanic_' + str(nb_vals) + '_synth', compute=True, output_path='../saves')

    #         G = explainer.build_attack_graph(compute=True, display_graph=False)
    #         print('total args:', len(G.nodes()))
    #         print('edges per node:', np.mean([len(G.edges(n)) for n in G.nodes()]))



