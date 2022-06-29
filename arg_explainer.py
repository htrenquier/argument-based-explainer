from distutils import extension
import matplotlib.pyplot as plt
from os import path
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations

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
        self.naive_extensions = None
        ## Current strategy:
        self.extension_strategy = 'naive'
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
            minimals = pd.read_pickle(path.join(output_path, self.exp_name + '_minimals.df'))
            covi_by_arg = pd.read_pickle(path.join(output_path, self.exp_name + '_covibyarg.df'))
            covc_by_arg = pd.read_pickle(path.join(output_path, self.exp_name + '_covcbyarg.df'))
        
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

        n = 0
        minimals = None
        while not minimals or len(minimals[0][-1]) != 0 or len(minimals[1][-1]) != 0:
            n += 1
            args, minimals = generate_args_lenN(n, instances_by_feature, X.toarray(), y, minimals)
            print("len ", n, ":", len(minimals[0][n-1]), ', ', len(minimals[1][n-1]))
        
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
        for cl in range(2):
            for l in range(len(minimals[cl])):
                for h1 in minimals[cl][l]:
                    for l2 in range(l-1):
                        for h2 in minimals[1-cl][l2]:
                            if self.consistent(h1, h2):
                                R_atk.append((h1, h2))
        print(self.exp_name)
        pp = path.join(self.output_path, self.exp_name + '_R_atk.df')
        pd.to_pickle(R_atk, pp)


    def evaluate_r_atk(self, minimals):
        self.build_r_atk(minimals)
        R_atk = pd.read_pickle(path.join(self.output_path, self.exp_name + '_R_atk.df'))
        print('len', len(R_atk))

        G = nx.Graph()
        G.add_edges_from(R_atk)
        #G = nx.petersen_graph()
        nx.draw(G, with_labels=True, font_weight='bold')
        nx.drawing.nx_pydot.write_dot(G, "R_atk_fig.dot")
        plt.savefig("R_atk_fig.png")
        return G

    def build_attack_graph(self):
        """
        """
        # Building Attack graph
        G = self.evaluate_r_atk(self.minimals)

        degs = np.array(list(G.degree()), dtype = [('node', 'object'), ('degree', int)])
        degrees = np.sort(degs, order='degree')
        print(degrees[-20:])

    def build_naive_extensions(self):
        """
        """
        # Building naive extensions
        R_atk = pd.read_pickle(path.join(self.output_path, self.exp_name + '_R_atk.df'))

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

    def strategy(self, strategy='max_covi'):
        # TODO: implement strategy
        """
        Returns the extension to be used for explanations.
        """
        if strategy == 'max_covi':
            covi_by_ext = dict()
            covc_by_ext = dict()
            max_covi = {}
            max_covi_ext = {}

            for ext in self.naive_extensions.values():
                covi = set.union(*[self.covi_by_arg[arg] for arg in ext])
                covi_by_ext.update({frozenset(ext): covi})
                if len(covi) > len(max_covi):
                    max_covi_ext = ext
                    max_covi = covi
                covc = set.union(*[self.covc_by_arg[arg] for arg in ext])
                covc_by_ext.update({frozenset(ext): covc})

            print(len(covi_by_ext[frozenset(max_covi_ext)]), "/", len(self.X))

            sorted_covs = [len(cov) for cov in covi_by_ext.values()]
            sorted_covs.sort()
            self.covi_by_extension = covi_by_ext
            self.covc_by_extension = covc_by_ext
            self.extension_strategy = max_covi_ext


    def explain(self, instance, i):
        ext_ = self.extension_strategy
        expl = set()
        instance_set = set(np.where(instance.toarray() != 0)[1])
        if i in self.covi_by_extension[frozenset(ext_)]:
            for arg in ext_:
                if arg.issubset(instance_set):
                    expl.add(arg)
        return expl

    def parse_features(self, explanation):
        parsed_expl = set()
        for arg in explanation:
            parsed_expl.add(frozenset([self.feature_names[k] for k in arg]))
        return parsed_expl

    def display_explanations(self, slice='all', strategy='max_covi'):
        if slice == 'all':
            slice = range(self.X.shape[0])
        else:
            slice = slice
        empty = 0
        tot = 0
        for k in slice:
            expl = self.parse_features(self.explain(self.X.getrow(k), k, strategy=strategy))
            print(k, expl)
            if not expl:
                empty += 1
            tot += 1

        print(empty, tot)