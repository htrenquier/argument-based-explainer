from distutils import extension
import matplotlib.pyplot as plt
from os import path
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations
import time
import io
import os

class ArgTabularExplainer(object):
    """

    """
    def __init__(self, model, dataset, y, exp_name, compute=False, output_path='saves') -> None:
        self.model = model
        self.dataset = dataset
        assert len(np.unique(np.array(y))) == 2
        self.y = y
        self.exp_name = exp_name if exp_name else 'arg-exp_default_' + str(len(y))
        self.output_path = output_path
        self.oh_enc = OneHotEncoder(handle_unknown='ignore', sparse=True)
        self.X = self.oh_enc.fit_transform(self.dataset).todok()
        self.feature_names = self.oh_enc.get_feature_names_out(dataset.columns)
        self.t_X = self.X.transpose().toarray()
        self.G = None
        self.node_dict = None # in case of extraction for 3rd party use
        ## Current strategy:
        self.strategy = {'selection' : None,
                         'inference': None,
                         'explanation_set': None,
                         'covi' : None,
                         'covc' : None,
                         'temp_cov': None}
        self.covi_by_extension = None
        self.covc_by_extension = None
        self.arg_by_instance = dict()
        self.instance_by_arg = dict()

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
                            self.arg_by_instance.update({frozenset(potential_arg): selection}) #arg by instance
                            self.instance_by_arg.update({frozenset(selection): set(potential_arg)}) #instance by arg
                            
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
        return self.G
    
    def export_graph(self, file_type, output_path='saves'):

        def graph_to_tgf(G, file_path):
            tgf_file = file_path + '.tgf'
            node_dict = {}
            i_ = 0
            with open(tgf_file, 'w') as f:
                for n in G.nodes():
                    i_ += 1
                    node_dict[n] = i_
                    f.write(str(i_)+ '\n')
                f.write('#\n')
                for u, v in G.edges():
                    f.write(str(node_dict[u]) + ' ' + str(node_dict[v]) + '\n')
                    f.write(str(node_dict[v]) + ' ' + str(node_dict[u]) + '\n')
            return node_dict
            
        def graph_to_asp(G, file_path):
            asp_file = file_path + '.asp'
            node_dict = {}
            i_ = 0
            with open(asp_file, 'w') as f:
                for n in G.nodes():
                    i_ += 1
                    node_dict[n] = i_
                    f.write("arg(" + str(i_) + ").\n")
                for u, v in G.edges():
                    f.write("att(" + str(node_dict[u]) + ',' + str(node_dict[v]) + ').\n')
                    f.write("att(" + str(node_dict[v]) + ',' + str(node_dict[u]) + ').\n')
            return node_dict
        
        file_path = path.join(output_path, self.exp_name)
        
        if file_type == 'tgf':
            self.node_dict = graph_to_tgf(self.G, file_path)
        elif file_type == 'asp':
            self.node_dict = graph_to_asp(self.G, file_path)
        else:
            print('file_type not recognized')

    def find_undefined_instances(self): 
        new_instances = []
        if self.G:
            hist = nx.degree_histogram(self.G)
            print('degree histogram:', hist)
            print('max degree:', max(hist))
            max_deg_node = sorted(self.G.degree, key=lambda x: x[1], reverse=True)
            attackers = nx.edges(max_deg_node[0])

            for a in attackers:
                continue
    
    def extension_generator_from_sat(self, file=None):
        if file is None:
            print('No file given')
        else:
            assert self.node_dict is not None # No arg_map generated after graph extraction
            arg_map = {v: k for k, v in self.node_dict.items()}
            
            try:
                # pre-processing ASP answer file
                if '_preprocessed' in file:
                    print("File already pre-processed")
                elif os.path.isfile(file[:-4] + '_preprocessed' + file[-4:]):
                    print("Using pre-processed file...")
                else:
                    with open(file, 'r') as f:
                        print('Pre-processing file...')
                        with open(file[:-4] + '_preprocessed' + file[-4:], 'w') as f2:
                            for k in range(3):
                                f.readline()
                            for line in f.readlines()[:-6]:
                                if 'Answer' not in line:
                                    f2.write(line.replace('in(', '').replace(')',''))
                    print('Done')
                
                file = file[:-4] + '_preprocessed' + file[-4:]
                print('Reading', file)
                #stream = io.open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True)
                #for line in stream.readlines():
                c = 0
                with open(file, 'r') as f:
                    for line in f.readlines():
                        c += 1
                        try:
                            e_ = np.array(line.split(' ')).astype(int)
                            yield([arg_map[e] for e in e_])
                        except Exception as e:
                            print("Skipped line no", c, ": ", line, e)
                    
            except IOError as e:
                print("IOError: file stream error", e)

    def extension_generator_from_graph(self):
        print("Working with NetworkX to find naive extensions:")
        print("Graph density = ", nx.density(self.G))
        print('Number of extensions: ', nx.graph_number_of_cliques(nx.complement(self.G)))
        return nx.find_cliques(nx.complement(self.G))
                
    def make_selection(self, alpha='max_covi', ext_generator=None):
        """
        Makes a selection in all naive extensions given by the generator
        alpha: the selection strategy
        """
        
        if self.strategy['selection'] is None:
            self.strategy['selection'] = []
        if ext_generator is None:
            ext_generator = self.extension_generator_from_graph()

        t0 = time.time()
        max_cov = 0
        max_cov_exts = []
            
        if alpha == 'max_covi':
            for ext in ext_generator:
                card_cov = len(set.union(*[self.covi_by_arg[arg] for arg in ext]))
                if card_cov > max_cov:
                    max_cov = card_cov
                    max_cov_exts = []
                if card_cov >= max_cov:
                    max_cov_exts.append(ext)
        
        elif alpha == 'max_covc':
            for ext in ext_generator:
                c += 1
                card_cov = len(set.union(*[self.covc_by_arg[arg] for arg in ext]))
                if card_cov > max_cov:
                    max_cov = card_cov
                    max_cov_exts = []
                if card_cov >= max_cov:
                    max_cov_exts.append(ext)
        
        else:
            print('Strategy not implemented')
            
        self.strategy['selection'].append(alpha)
        print("Time for selection: ", time.time()-t0)
        print("Len max_cov_exts: ", len(max_cov_exts))

        if self.covi_by_extension is None:
            self.covi_by_extension = dict()
            for ext in max_cov_exts:
                covi = set.union(*[self.covi_by_arg[arg] for arg in ext])
                self.covi_by_extension.update({frozenset(ext): covi})
            # TO VERIFY: strategy['covi'] should contain covi for all extensions even after multiple selections
        self.strategy['covi'] = set.union(*[self.covi_by_extension[frozenset(ext)] for ext in max_cov_exts])
        
        return max_cov_exts
            
    def apply_inference(self, extension_set, beta='universal'):
        """
        beta: inference strategy: 'universal', 'existence' or 'one'
        """
        
        if beta == 'universal':
            self.strategy['explanation_set'] = set.intersection(*[set(s) for s in extension_set])
        elif beta == 'existence':
            self.strategy['explanation_set'] = set.union(*[set(s) for s in extension_set])
        elif beta == 'one':
            self.strategy['explanation_set'] = set(next(iter(extension_set)))
        else:
            print(beta, "not Implemented")
        
        self.strategy['inference'] = beta
        return self.strategy['explanation_set']
        
    
    def build_naive_extensions(self, method='cliques', file=None):
        """
        DEPRECATED
        method: 'cliques' : uses the networkx library to return a generator of all maximal 
                            cliques of the complement graph of G
                 or 'solver': uses
        """
        # Building naive extensions
        # R_atk = pd.read_pickle(path.join(self.output_path, self.exp_name + '_R_atk.df'))

                # Compute naive extensions (or maximal anti-cliques: ac) only works with small graphs
        if method == 'cliques':
            if file is not None:
                print("No file needed for this method")
            print("Graph density = ", nx.density(self.G))
            print('Number of extensions: ', nx.graph_number_of_cliques(nx.complement(self.G)))
            return nx.find_cliques(nx.complement(self.G))
        elif method == 'solver':
            if file is None:
                print("No file provided")
                return 0
            try:
                # pre-processing file
                if '_preprocessed' in file:
                    print("File already pre-processed")
                elif os.path.isfile(file[:-4] + '_preprocessed' + file[-4:]):
                    print("Using pre-processed file...")
                else:
                    with open(file, 'r') as f:
                        print('Pre-processing file...')
                        with open(file[:-4] + '_preprocessed' + file[-4:], 'w') as f2:
                            for k in range(3):
                                f.readline()
                            for line in f.readlines()[:-6]:
                                if 'Answer' not in line:
                                    f2.write(line.replace('in(', '').replace(')',''))
                                    
                    print('Done')
                file = file[:-4] + '_preprocessed' + file[-4:]
                print('Streaming ', file)
                stream = io.open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True)
                for line in stream.readlines():
                    try:
                        yield(np.array(line.split(' ')).astype(int))
                    except:
                        print("Skipped line no", c, ": ", l)
                    
            except IOError:
                print("IOError: file stream error")
        else:
            print("Method not implemented")


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
        
        t_ne = 0
        t0 = time.time()
        t1 = time.time()
        for _ext in ne:
            t_ne += time.time() - t1
            nb_ne += 1
            ext = set(_ext)
            covi = set.union(*[self.covi_by_arg[arg] for arg in ext])
            covc = set.union(*[self.covc_by_arg[arg] for arg in ext])
            self.covi_by_extension.update({frozenset(ext): covi})
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
            t1 = time.time()

        print('Time spent on naive extensions:', t_ne, 's', '(', time.time() -  t0, ')')
                
        self.strategy['selection'] = selection
        self.strategy['inference'] = inference
        print('len(max_cov_exts)=', len(max_cov_exts), '/', nb_ne)
        if inference == 'universal':
            self.strategy['explanation_set'] = set.intersection(*max_cov_exts)
        elif inference == 'existence':
            self.strategy['explanation_set'] = set.union(*max_cov_exts)
        elif inference == 'one':
            self.strategy['explanation_set'] = max_cov_exts[0]
        
        self.strategy['covi'] = set.union(*[self.covi_by_extension[frozenset(ext)] for ext in max_cov_exts])
        self.strategy['covc'] = set.union(*[self.covc_by_extension[frozenset(ext)] for ext in max_cov_exts])
        
        if selection == 'max_covi':
            print('Covi strategy\'s coverage:', len(self.strategy['covi']))
            sorted_covs = [len(cov) for cov in self.covi_by_extension.values()]
        elif selection == 'max_covc':
            print('Covc strategy\'s coverage:', len(self.strategy['covc']))
            sorted_covs = [len(cov) for cov in self.covc_by_extension.values()]
        
        sorted_covs.sort()
        print('Top 5 covs:', sorted_covs[-5:])

    def explain(self, i):
        expl_set = self.strategy['explanation_set']
        cov_by_arg = None
        if self.strategy['selection'][-1] == 'max_covi':
            cov_by_arg = self.covi_by_arg
        elif self.strategy['selection'][-1] == 'max_covc':
            cov_by_arg = self.covc_by_arg
        assert cov_by_arg
        cov = set()
        expl = set()
        
        instance_set = set(np.where(self.X[i].toarray() != 0)[1])
        if i in self.strategy['covi']:
            for arg in expl_set:
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
        if not self.strategy['selection']:
            print('Defaulting to max_covi strategy.')
            self.set_strategy('max_covi', 'universal')
            # change to new process
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