from distutils import extension
import matplotlib.pyplot as plt
from os import path
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations
import copy

class ArgTabularExplainer(object):
    """

    """
    def __init__(self, model, dataset, y, exp_name, compute=False, output_path='saves') -> None:
        self.model = model
        self.dataset = dataset
        self.y = y
        self.exp_name = exp_name if exp_name else 'arg-exp_default_' + str(len(y))
        self.output_path = output_path
        self.oh_enc = OneHotEncoder(handle_unknown='ignore', sparse=True)
        self.X = self.oh_enc.fit_transform(self.dataset).todok()
        self.feature_names = self.oh_enc.get_feature_names_out(dataset.columns)
        self.t_X = self.X.transpose().toarray()
        self.G = None
        ## Current strategy:
        self.strategy = {'selection' : None,
                         'inference': None,
                         'explanation_set': None,
                         'temp_cov': None}
        self.covi_by_extension = None
        self.covc_by_extension = None

        self.ibyf, self.features_p_col, self.col_p_feature = self.preprocess_structures(
            self.dataset, self.t_X, self.feature_names)

        if compute:
            self.minimals, self.covi_by_arg, self.covc_by_arg = self.generate_args(self.ibyf, self.X, self.y)
            ## Save
            pp = path.join(output_path, self.exp_name + '_minimals.df')
            print('Saving to ', pp)
            pd.to_pickle(self.minimals, path.join(output_path, self.exp_name + '_minimals.df'))
            ### Cov
            pd.to_pickle(self.covi_by_arg, path.join(output_path, self.exp_name + '_covibyarg.df'))
            pd.to_pickle(self.covc_by_arg, path.join(output_path, self.exp_name + '_covcbyarg.df'))
        else: 
            ## Load
            self.minimals = pd.read_pickle(path.join(output_path, self.exp_name + '_minimals.df'))
            self.covi_by_arg = pd.read_pickle(path.join(output_path, self.exp_name + '_covibyarg.df'))
            self.covc_by_arg = pd.read_pickle(path.join(output_path, self.exp_name + '_covcbyarg.df'))
        
        self.arguments = self.read_args(self.minimals, self.feature_names)
        
    def preprocess_structures(self, dataset, t_X, feature_names):
        instances_by_feature = dict()
        for i, col in enumerate(t_X):
            instances_by_feature.update({i: list(np.where(col)[0])})
        
        features_p_col = {}
        col_p_feature = {}

        for col in dataset.columns:
            features_p_col[col] = set()
            for i, f in enumerate(feature_names):
                if col in f:
                    features_p_col[col].add(i)
                    col_p_feature[i] = col

        return instances_by_feature, features_p_col, col_p_feature

    def generate_args(self, instances_by_feature, X, y):
        """
        """
        # Coverage
        covi_by_arg = dict()
        covc_by_arg = dict()

        def generate_args_lenN(n, ibyf, dataset, predictions, minimals=None):
            """
            Generates arguments of length n, given arguments of length 1.. n-1
            :param n: length of arguments to be generated
            :param ibyf: instances_by_feature
            :param predictions:
            :param minimals: arguments (minimal)
            :return:
            """

            def is_minimal(potential_arg, cl, minimals, n):
                # cl is class
                set_potential_arg = set(potential_arg)
                for k in range(n):
                    for comb_ in combinations(potential_arg, k+1):
                        if frozenset(comb_) in minimals[cl][k]:
                            return False
                return True

            if minimals is None:
                minimals = ([], [])
            assert len(minimals[0]) == n-1
            minimals[0].append(set())
            minimals[1].append(set())

            args = [set(), set()]
            potential_args_checked_count = 0
            for i, row in enumerate(dataset):
                for potential_arg in combinations(np.where(row)[0], n):
                    cl = predictions[i]
                    potential_args_checked_count += 1
                    if not is_minimal(potential_arg, cl, minimals, n-1):
                        continue
                    selection = set.intersection(*[set(ibyf[w]) for w in potential_arg])  # all rows with all features of potential argument
                    selection_preds = [predictions[i_] for i_ in selection]
                    if selection_preds[:-1] == selection_preds[1:]:
                            args[selection_preds[0]].add(frozenset(potential_arg))
                            covi_by_arg.update({frozenset(potential_arg): selection}) #covi
                            minimals[cl][n-1].add(frozenset(potential_arg))
                            covc_by_arg.update({frozenset(potential_arg): set(selection_preds)}) #covc
            print(potential_args_checked_count, ' potential arg checked.')
            return args, minimals

        print('Generating arguments')
        n = 0
        minimals = None
        special = False
        while not minimals or len(minimals[0][-1]) != 0 or len(minimals[1][-1]) != 0  or special:
            special = False
            n += 1
            args, minimals = generate_args_lenN(n, instances_by_feature, X.toarray(), y, minimals)
            print("len ", n, ":", len(minimals[0][n-1]), ', ', len(minimals[1][n-1]))
            if n==1 and ( not minimals[0][0] and not minimals[1][0]):
                special = True
        
        return minimals, covi_by_arg, covc_by_arg

    def read_args(self, minimals, feature_names):
        arguments = [[], []]
        for cl in range(len(minimals)):
            for a in range(len(minimals[cl])):
                for f in minimals[cl][a]:
                    arguments[cl].append(tuple([feature_names[k] for k in f]))
        return arguments

    def consistent(self, arg1, arg2):
        for f1, f2 in zip(list(arg1), list(arg2)):
            if f1 != f2 and self.col_p_feature[f1] == self.col_p_feature[f2]:
                return False
        return True

    def build_r_atk(self, minimals):
        R_atk = []
        all_args = []
        for cl in range(2):
            all_args.append(list(set.union(*list(minimals[cl]))))
        print(len(all_args[0]), len(all_args[1]), " args total")
        for a1 in all_args[0]:
            for a2 in all_args[1]:
                if self.consistent(a1,a2):
                    R_atk.append((a1,a2))
        """
        for cl in range(2):
            nb_added = 0
            for l in range(len(minimals[cl])):
                for h1 in minimals[cl][l]:
                    for l2 in range(l-1):
                        for h2 in minimals[1-cl][l2]:
                            if self.consistent(h1, h2):
                                R_atk.append((h1, h2))
                                nb_added += 1
            print(nb_added, " attacks added")
        """
        print(self.exp_name)
        pp_atk = path.join(self.output_path, self.exp_name + '_R_atk.df')
        pp_aa = path.join(self.output_path, self.exp_name + '_all_args.df')
        pd.to_pickle(R_atk, pp_atk)
        pd.to_pickle(all_args, pp_aa)
        return all_args, R_atk
        

    def build_attack_graph(self, compute=False, display_graph=False):
        """
        """
        # Building Attack graph
        pp_aG = path.join(self.output_path, self.exp_name + '_atk_graph.df')
        
        if compute:
            all_args, R_atk = self.build_r_atk(self.minimals)
            self.G = nx.Graph()
            self.G.add_nodes_from(np.concatenate(all_args))
            self.G.add_edges_from(R_atk)
            pd.to_pickle(self.G, pp_aG)
        else:
            R_atk = pd.read_pickle(path.join(self.output_path, self.exp_name + '_R_atk.df'))
            #all_args = pd.read_pickle(path.join(self.output_path, self.exp_name + '_all_args.df'))
            self.G = pd.read_pickle(pp_aG)
        print('len(R_atk) = ', len(R_atk))
        
        if display_graph:
            nx.draw(self.G, with_labels=False)
            nx.drawing.nx_pydot.write_dot(self.G,path.join(self.output_path, self.exp_name + "G_atk_fig.dot"))
            plt.savefig(path.join(self.output_path, self.exp_name  +"G_atk_fig.png"))
        
        degs = np.array(list(self.G.degree()), dtype = [('node', 'object'), ('degree', int)])
        degrees = np.sort(degs, order='degree')
        print('5 highest degrees:', degrees[-5:])
        print('5 lowest degrees:', degrees[:5])
        
    
    def build_naive_extensions(self):
        """
        """
        # Building naive extensions
        # R_atk = pd.read_pickle(path.join(self.output_path, self.exp_name + '_R_atk.df'))

                # Compute naive extensions (or maximal anti-cliques: ac)
        print(nx.density(self.G))
        return nx.find_cliques(nx.complement(self.G))
        """
        all_args = set()
        for cl in range(len(self.minimals)):
            for l in range(len(self.minimals[cl])):
                all_args.update(self.minimals[cl][l])

        print(len(all_args), ' args in total.')

        # Finding naive extensions can also be done by finding all maximal independent 
        # sets: nx.maximal_independent_set(G) can output one random one.

        naive_extensions = {}
        for (h1, h2) in R_atk:
            if h1 not in naive_extensions:
                naive_extensions[h1] = all_args.copy()
            if h2 not in naive_extensions:
                naive_extensions[h2] = all_args.copy()
            naive_extensions[h1].discard(h2)
            naive_extensions[h2].discard(h1)
        self.naive_extensions = naive_extensions
        """

    def set_strategy(self, selection='max_covi', inference='universal'):
        # TODO: implement strategies
        """
        Returns the extension to be used for explanations.
        """
       
        self.covi_by_extension = dict()
        self.covc_by_extension = dict()
        covi = set()
        covc = set()
        max_cov = set()
        max_cov_exts = None
        # Reset naive_extensions generator
        ne = self.build_naive_extensions()
        assert ne
        nb_ne = 0
         
        for _ext in ne:
            nb_ne += 1
            ext = set(_ext)
            covi = set.union(*[self.covi_by_arg[arg] for arg in ext])
            self.covi_by_extension.update({frozenset(ext): covi})
            covc = set.union(*[self.covc_by_arg[arg] for arg in ext])
            self.covc_by_extension.update({frozenset(ext): covc})
            
            if selection == 'max_covi':
                cov = covi
            elif selection == 'max_covc':
                cov = covc
            
            if len(cov) > len(max_cov):
                max_cov = cov
                max_cov_exts = [ext]
            elif len(cov) == len(max_cov):
                max_cov_exts.append(ext)
                
        self.strategy['selection'] = selection
        self.strategy['inference'] = inference
        print('len(max_cov_exts)=',len(max_cov_exts), '/', nb_ne)
        if inference == 'universal':
            self.strategy['explanation_set'] = set.intersection(*max_cov_exts)
        elif inference == 'existence':
            self.strategy['explanation_set'] = set.union(*max_cov_exts)
        elif inference == 'one':
            self.strategy['explanation_set'] = max_cov_exts[0]
        
        if selection == 'max_covi':
            self.strategy['covi'] = set.union(*[self.covi_by_extension[frozenset(ext)] for ext in max_cov_exts])
            print('Covi strategy\'s coverage:', len(self.strategy['covi']))
            sorted_covs = [len(cov) for cov in self.covi_by_extension.values()]
        elif selection == 'max_covc':
            self.strategy['covc'] = set.union(*[self.covc_by_extension[frozenset(ext)] for ext in max_cov_exts])
            print('Covc strategy\'s coverage:', len(self.strategy['covc']))
            sorted_covs = [len(cov) for cov in self.covc_by_extension.values()]
        
        sorted_covs.sort()
        print('Top 5 covs:', sorted_covs[-5:])


    def explain(self, i):
        ext_ = self.strategy['explanation_set']
        cov_by_arg = None
        if self.strategy['selection'] == 'max_covi':
            cov_by_arg = self.covi_by_arg
        elif self.strategy['selection'] == 'max_covc':
            cov_by_arg = self.covc_by_arg
        assert cov_by_arg
        cov = set()
        expl = set()
        
        instance_set = set(np.where(self.X[i].toarray() != 0)[1])
        if i in self.covi_by_extension[frozenset(ext_)]:
            for arg in ext_:
                if arg.issubset(instance_set):
                    expl.add(arg)
                    cov.update(cov_by_arg[arg])
        return expl, cov

    def parse_features(self, explanation):
        parsed_expl = set()
        for arg in explanation:
            parsed_expl.add(frozenset([self.feature_names[k] for k in arg]))
        return parsed_expl

    def display_explanations(self, slice='all'):
        if not self.strategy:
            print('Defaulting to max_covi strategy.')
            self.set_strategy('max_covi')
        if slice == 'all':
            slice = range(self.X.shape[0])
        else:
            slice = slice
        empty = 0
        tot = 0
        for k in slice:
            expl, cov = self.explain(k)
            expl_parsed = self.parse_features(expl)
            example = next(iter(expl_parsed)) if expl_parsed else None
            print('id:', k, 'coverage:', len(cov), 'Arg 1/' + str(len(expl_parsed)) + ':', example)
            if not expl:
                empty += 1
            tot += 1

        print(empty, tot)
        
    def explain_instance(self, k):
        expl, cov = self.explain(k)
        expl_parsed = self.parse_features(expl)
        print('id:', k, 'coverage:', len(cov), 'Args' + str(len(expl_parsed)) + '/' + str(len(expl_parsed)) + ':', expl_parsed)