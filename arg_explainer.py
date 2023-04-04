from distutils import extension
import matplotlib.pyplot as plt
from os import path
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations, product
import time
import io
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool
# from multiprocessing import Pool


class ArgTabularExplainer(object):
    # TODO: separate init and compute
    """

    """
    def __init__(self, dataset_manager, exp_name, compute=False, output_path='saves', verbose=True) -> None:
        self.dm = dataset_manager
        self.verbose = verbose

        self.exp_name = exp_name if exp_name else 'arg-exp_default_' + str(len(self.dm.nb_classes))
        self.output_path = output_path

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

        print(self.exp_name)
        if compute:
            self.minimals, self.covi_by_arg, self.covc_by_arg = self.generate_args(self.dm.ibyfv, self.dm.X, self.dm.predictions)
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
        
        self.arguments = self.read_args(self.minimals, self.dm.feature_value_names)
        

    def target(self, i, row, args_checked, minimals, ibyfv, predictions, n):
        
        def is_minimal(potential_arg, cl, minimals, n):
            # cl is class
            for k in range(n):
                for comb_ in combinations(potential_arg, k+1):
                    if frozenset(comb_) in minimals[cl]:
                        return False
            return True
        
        arg_batch = list()
        for potential_arg in combinations(np.nonzero(row)[0], n):
            if potential_arg in args_checked:
                continue
            args_checked.append(potential_arg)
            cl = predictions[i]

            if not is_minimal(potential_arg, cl, minimals, n-1):
                continue
            
            selection = set.intersection(*[ibyfv[w] for w in potential_arg])  # all rows with all features of potential argument
            selection_preds = [predictions[i_] for i_ in selection]
            if selection_preds[:-1] == selection_preds[1:]:
                arg_batch.append((frozenset(potential_arg), selection, selection_preds))
        return arg_batch

    

    def generate_args_lenN_mp(self, n, ibyfv, X_enc, predictions, minimals, covi_by_arg, covc_by_arg, verbose):
        """
        Generates arguments of length n, given arguments of length 1.. n-1
        :param n: length of arguments to be generated
        :param ibyfv: instances_by_feature_value 
        :param predictions:
        :param minimals: arguments (minimal)
        :return:
        """
        
        args = [set(), set()]
        minimals_count = 0
        not_minimal_count = 0

        mp_results = []
        manager = mp.Manager()
        args_checked = manager.list()

        with Pool(mp.cpu_count()) as pool:
            for i, row in enumerate(X_enc):
                mp_results.append(pool.apply_async(self.target,\
                                    args=(i, row, args_checked, minimals, ibyfv, predictions, n)))
            pool.close()
            pool.join()

        for res in mp_results:
            for arg in res.get():
                minimals_count += 1
                (potential_arg, selection, selection_preds) = arg
                if selection_preds[:-1] == selection_preds[1:]:
                    cl = selection_preds[0]
                    args[cl].add(potential_arg)
                    minimals[cl].add(potential_arg)
                    covi_by_arg.update({potential_arg: selection}) #covi
                    covc_by_arg.update({potential_arg: set(selection_preds)}) #covc
                    self.arg_by_instance.update({potential_arg: selection}) #arg by instance
                    self.instance_by_arg.update({frozenset(selection): set(potential_arg)}) #instance by arg
            else:
                not_minimal_count += 1

        if verbose:
            print("len ", n, ":", len(args[0]), ', ', len(args[1]))
            print(minimals_count, 'potential arg checked (',
                            not_minimal_count, 'not minimal) - mp')
        return args, minimals
        
                
    def generate_args_lenN(self, n, ibyfv, X_enc, predictions, minimals, covi_by_arg, covc_by_arg, verbose):
        """
        Generates arguments of length n, given arguments of length 1.. n-1
        :param n: length of arguments to be generated
        :param ibyfv: instances_by_feature_value 
        :param predictions:
        :param minimals: arguments (minimal)
        :return:
        """
        def is_minimal(potential_arg, cl, minimals, n):
            # cl is class
            for k in range(n):
                for comb_ in combinations(potential_arg, k+1):
                    if frozenset(comb_) in minimals[cl]:
                        return False
            return True

        args = [set(), set()]
        minimals_count = 0
        not_minimal_count = 0
        arg_count = 0
        args_checked = set()

        for i, row in enumerate(X_enc):
            for potential_arg in combinations(np.nonzero(row)[0], n):
                #potential_arg = sorted(potential_arg)
                if potential_arg in args_checked:
                    continue

                args_checked.add(potential_arg)
                cl = predictions[i]
                
                if not is_minimal(potential_arg, cl, minimals, n-1):
                    not_minimal_count += 1
                    continue

                minimals_count += 1
                
                selection = set.intersection(*[ibyfv[w] for w in potential_arg])  # all rows with all features of potential argument
                selection_preds = [predictions[i_] for i_ in selection]

                if selection_preds[:-1] == selection_preds[1:]:
                        arg_count += 1
                        args[selection_preds[0]].add(frozenset(potential_arg))
                        covi_by_arg.update({frozenset(potential_arg): selection}) #covi
                        minimals[cl].add(frozenset(potential_arg))
                        covc_by_arg.update({frozenset(potential_arg): set(selection_preds)}) #covc
                        self.arg_by_instance.update({frozenset(potential_arg): selection}) #arg by instance
                        self.instance_by_arg.update({frozenset(selection): set(potential_arg)}) #instance by arg


        if verbose:
            print("len ", n, ":", len(args[0]), ', ', len(args[1]))
            print(minimals_count, 'potential arg checked (',
                            not_minimal_count, 'not minimal)')
        return args, minimals

    def generate_args(self, instances_by_feature, X, y):
        """
        """
        # Coverage
        covi_by_arg = dict()
        covc_by_arg = dict()

        minimals = (set(), set())

        if self.verbose:
            print('Generating arguments')
        for n in range(1, len(self.dm.feature_names) + 1):
            args, minimals = self.generate_args_lenN_mp(n, instances_by_feature, X.toarray(), y, minimals, covi_by_arg, covc_by_arg, self.verbose)

        print('Total number of arguments: ', len(minimals[0]) + len(minimals[1]))
        return minimals, covi_by_arg, covc_by_arg
    

    def read_args(self, minimals, feature_value_names):
        arguments = [[], []]
        for cl in range(len(minimals)):
                for f in minimals[cl]:
                    arguments[cl].append(tuple([feature_value_names[k] for k in f]))
        return arguments


    def consistent(self, arg1, arg2):
        """
          Pre condition: arg1 and arg2 are subsets of a possible instance
          returns True if arg1 and arg2 are consistent
        """
        u = sorted(list(set.union(set(arg1), set(arg2))))
        for k in range(len(u)-1):
            if self.dm.feature_p_value[u[k]] == self.dm.feature_p_value[u[k+1]]:
                return False
        return True


    def build_r_atk(self, minimals):
        R_atk = []
        all_args = []
        for cl in range(2):
            all_args.append(list(minimals[cl]))

        for a1, a2 in product(minimals[0], minimals[1]):
            if self.consistent(a1,a2):
                R_atk.append((a1,a2))

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

        if display_graph:
            nx.draw(self.G, with_labels=False)
            nx.drawing.nx_pydot.write_dot(self.G,path.join(self.output_path, self.exp_name + "G_atk_fig.dot"))
            plt.savefig(path.join(self.output_path, self.exp_name  + "G_atk_fig.png"))
        
        return self.G
    
    def af_analysis(self, remove=[]):
        aG_files = [f for f in os.listdir(self.output_path)\
            if path.isfile(path.join(self.output_path, f))\
                and f.endswith('atk_graph.df')\
                and f.startswith(self.exp_name.split('_')[0])]
        R_atk_files = [f for f in os.listdir(self.output_path)\
            if path.isfile(path.join(self.output_path, f))\
                and f.endswith('R_atk.df')\
                and f.startswith(self.exp_name.split('_')[0])]
        minimals_files = [f for f in os.listdir(self.output_path)\
            if path.isfile(path.join(self.output_path, f))\
                and f.endswith('minimals.df')\
                and f.startswith(self.exp_name.split('_')[0])]
        
        file_groups = [aG_files, R_atk_files, minimals_files]
        cleans = [[]] * len(file_groups)
        
        def remove_file_by_kw(file_group, remove):
            cleans = []
            for f in file_group:
                clean = True
                for kw in remove:
                    if kw in f:
                        clean = False
                        break
                if clean:
                    cleans.append(f)
            return cleans

        aG_files = remove_file_by_kw(aG_files, remove)
        R_atk_files = remove_file_by_kw(R_atk_files, remove)
        minimals_files = remove_file_by_kw(minimals_files, remove)
        
        list_R_atk = []
        list_degs = []
        for f in R_atk_files:
            list_R_atk.append(len(pd.read_pickle(path.join(self.output_path, f))))
        print(list_R_atk)

        nargs_list = []
        natk_list = []
        ninst_list = []
        list_coherences = []
        for f in aG_files:
            G_ = pd.read_pickle(path.join(self.output_path, f))
            ninst_list.append(int(f.split("_")[1]))
            nargs_list.append(len(G_.nodes()))
            natk_list.append(len(G_.edges()))
            degs = [d for n, d in G_.degree()]
            list_degs.append(degs)
            list_coherences.append(1 - (np.count_nonzero(degs)/len(degs)))

        ninst_list_sorted, nargs_list_sorted = zip(*sorted(zip(ninst_list, nargs_list)))
        print('ninst:', ninst_list_sorted)
        print('nargs:',nargs_list_sorted)
        plt.plot(ninst_list_sorted, nargs_list_sorted)
        plt.xlabel("# instances")
        plt.ylabel("# arguments")
        plt.show()

        ninst_list_sorted, natk_list_sorted = zip(*sorted(zip(ninst_list, natk_list)))
        print('ninst:', ninst_list_sorted)
        print('natk:', natk_list_sorted)
        plt.plot(ninst_list_sorted, natk_list_sorted)
        plt.xlabel("# instances")
        plt.ylabel("# attacks")
        plt.show()

        nargs_list_sorted, natk_list_sorted = zip(*sorted(zip(nargs_list, natk_list)))
        print('nargs:', nargs_list_sorted)
        print('ninst:', natk_list_sorted)
        plt.plot(nargs_list_sorted, natk_list_sorted)
        plt.xlabel("# arguments")
        plt.ylabel("# attacks")
        plt.show()

        for f in minimals_files:
            minimals = pd.read_pickle(path.join(self.output_path, f))
            arg_lengths = [0] * (len(self.dm.feature_names) + 1)
            for arg in minimals[0]:
                arg_lengths[len(arg)] += 1
            for arg in minimals[1]:
                arg_lengths[len(arg)] += 1
            plt.plot(arg_lengths, label = f.split("_")[1] + " instances")
            
        plt.xlabel("arguments' length")
        plt.ylabel("# arguments")
        plt.legend()
        plt.show()

        fig = plt.figure(figsize =(10, 7))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.boxplot(list_degs)
        plt.show()

        list_coherences_sorted, ninst_list_sorted = zip(*sorted(zip(ninst_list, list_coherences), reverse=True))
        print(list_coherences_sorted, ninst_list_sorted)

        plt.plot(list_coherences_sorted, ninst_list_sorted)
        plt.xlabel("# instances")
        plt.ylabel("% coherent arguments")
        plt.show()

        return
        R_atk = pd.read_pickle(pp_Ratk)
        self.G = pd.read_pickle(pp_aG)
        print('len(R_atk) = ', len(R_atk))
        
        degs = np.array(list(self.G.degree()), dtype = [('node', 'object'), ('degree', int)])
        degrees = np.sort(degs, order='degree')
        print('5 highest degrees:', degrees[-5:])
        print('5 lowest degrees:', degrees[:5])

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
                    #f.write("att(" + str(node_dict[v]) + ',' + str(node_dict[u]) + ').\n')
            return node_dict
        
        file_path = path.join(output_path, self.exp_name)
        
        if file_type == 'tgf':
            self.node_dict = graph_to_tgf(self.G, file_path)
        elif file_type == 'asp':
            self.node_dict = graph_to_asp(self.G, file_path)
        else:
            print('file_type not recognized')

    # def find_undefined_instances(self): 
    #     new_instances = []
    #     if self.G:
    #         hist = nx.degree_histogram(self.G)
    #         print('degree histogram:', hist)
    #         print('max degree:', max(hist))
    #         max_deg_node = sorted(self.G.degree, key=lambda x: x[1], reverse=True)
    #         attackers = nx.edges(max_deg_node[0])

    #         for a in attackers:
    #             continue
    
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
                
    def make_selection(self, alpha='max_covi', ext_generator=None, ext_cap=-1):
        """
        Makes a selection in all naive extensions given by the generator
        alpha: the selection strategy
        """
        
        if self.strategy['selection'] is None:
            self.strategy['selection'] = []
        if ext_generator is None:
            ext_generator = self.extension_generator_from_graph()

        t0 = time.time()
        max_ = 0
        exts_ = []
        c = 0

        if alpha == 'max_covi':
            for ext in ext_generator:
                if ext_cap > 0 and c >= ext_cap:
                    break
                card_cov = len(set.union(*[self.covi_by_arg[arg] for arg in ext]))
                if card_cov > max_:
                    max_ = card_cov
                    exts_ = []
                if card_cov >= max_:
                    exts_.append(ext)
                c += 1
        
        elif alpha == 'max_covc':
            for ext in ext_generator:
                if ext_cap > 0 and c >= ext_cap:
                    break
                card_cov = len(set.union(*[self.covc_by_arg[arg] for arg in ext]))
                if card_cov > max_:
                    max_ = card_cov
                    exts_ = []
                if card_cov >= max_:
                    exts_.append(ext)
                c += 1
        elif alpha == 'max_card':
            for ext in ext_generator:
                if ext_cap > 0 and c >= ext_cap:
                    break
                card_ext = len(ext)
                if card_ext > max_:
                    max_ = card_ext
                    exts_ = []
                if card_ext >= max_:
                    exts_.append(ext)
                c += 1
        elif alpha == 'max_covi_incl':
            covs_ = dict()
            for ext in ext_generator:
                if ext_cap > 0 and c >= ext_cap:
                    break
                cov_ = set.union(*[self.covi_by_arg[arg] for arg in ext])
                to_remove = []
                add_ext = True
                for k, c_ in covs_.items(): # check if the extension is included in a previous one
                    if c_.issubset(cov_):
                        to_remove.append(k)
                    elif cov_.issubset(c_):
                        add_ext = False
                for k in to_remove:
                    covs_.pop(k)  
                if add_ext:
                    covs_.update({frozenset(ext) : cov_})
                c += 1
            exts_ = [ext for ext in covs_.keys()]
        else:
            print('Strategy not implemented')
            
        self.strategy['selection'].append(alpha)
        print("Time for selection: ", time.time()-t0)
        print("Len max_cov_exts: ", len(exts_))

        if self.covi_by_extension is None:
            self.covi_by_extension = dict()
        for ext in exts_:
            covi = set.union(*[self.covi_by_arg[arg] for arg in ext])
            self.covi_by_extension.update({frozenset(ext): covi})
            # TO VERIFY: strategy['covi'] should contain covi for all extensions even after multiple selections
        self.strategy['covi'] = set.union(*[self.covi_by_extension[frozenset(ext)] for ext in exts_])
        
        return exts_
            
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
                 or 'solver': uses a file containing the output of the Aspartix (clingo) solver called as follows:
                 # clingo adultshort_100.asp naive.dl filter.lp 0 > adultshort_sat.txt
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


    # def set_strategy(self, selection='max_covi', inference='universal'):
    #     # TODO: implement strategies
    #     """
    #     DEPRECATED
    #     Returns the extension to be used for explanations.
    #     """
       
    #     self.covi_by_extension = dict()
    #     self.covc_by_extension = dict()
    #     covi = set()
    #     covc = set()
    #     max_cov = set()
    #     max_cov_exts = None
    #     # Reset naive_extensions generator
    #     ne = self.build_naive_extensions()
    #     assert ne
    #     nb_ne = 0
        
    #     t_ne = 0
    #     t0 = time.time()
    #     t1 = time.time()
    #     for _ext in ne:
    #         t_ne += time.time() - t1
    #         nb_ne += 1
    #         ext = set(_ext)
    #         covi = set.union(*[self.covi_by_arg[arg] for arg in ext])
    #         covc = set.union(*[self.covc_by_arg[arg] for arg in ext])
    #         self.covi_by_extension.update({frozenset(ext): covi})
    #         self.covc_by_extension.update({frozenset(ext): covc})
            
    #         if selection == 'max_covi':
    #             cov = covi
    #         elif selection == 'max_covc':
    #             cov = covc
            
    #         if len(cov) > len(max_cov):
    #             max_cov = cov
    #             max_cov_exts = [ext]
    #         elif len(cov) == len(max_cov):
    #             max_cov_exts.append(ext)
    #         t1 = time.time()

    #     print('Time spent on naive extensions:', t_ne, 's', '(', time.time() -  t0, ')')
                
    #     self.strategy['selection'] = selection
    #     self.strategy['inference'] = inference
    #     print('len(max_cov_exts)=', len(max_cov_exts), '/', nb_ne)
    #     if inference == 'universal':
    #         self.strategy['explanation_set'] = set.intersection(*max_cov_exts)
    #     elif inference == 'existence':
    #         self.strategy['explanation_set'] = set.union(*max_cov_exts)
    #     elif inference == 'one':
    #         self.strategy['explanation_set'] = max_cov_exts[0]
        
    #     self.strategy['covi'] = set.union(*[self.covi_by_extension[frozenset(ext)] for ext in max_cov_exts])
    #     self.strategy['covc'] = set.union(*[self.covc_by_extension[frozenset(ext)] for ext in max_cov_exts])
        
    #     if selection == 'max_covi':
    #         print('Covi strategy\'s coverage:', len(self.strategy['covi']))
    #         sorted_covs = [len(cov) for cov in self.covi_by_extension.values()]
    #     elif selection == 'max_covc':
    #         print('Covc strategy\'s coverage:', len(self.strategy['covc']))
    #         sorted_covs = [len(cov) for cov in self.covc_by_extension.values()]
        
    #     sorted_covs.sort()
    #     print('Top 5 covs:', sorted_covs[-5:])

    def explain(self, i):
        expl_set = self.strategy['explanation_set']
        cov_by_arg = None
        if self.strategy['selection'][-1] == 'max_covc':
            cov_by_arg = self.covc_by_arg
        else: # self.strategy['selection'][-1] == 'max_covi':
            cov_by_arg = self.covi_by_arg
        
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

    # def parse_features(self, explanation):
    #     parsed_expl = set()
    #     for arg in explanation:
    #         parsed_expl.add(frozenset([self.feature_names[k] for k in arg]))
    #     return parsed_expl

    def display_explanations(self, slice='all', verbose=True):
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
            if verbose:
                print('id:', k, 'coverage:', len(cov), 'Arg 1/' + str(len(expl_parsed)) + ':', example)
            if not expl:
                empty += 1
            tot += 1

        print('success = ', (tot-empty)/tot, '(empty=', empty, ')')
        
    def explain_instance(self, k):
        expl, cov = self.explain(k)
        expl_parsed = self.parse_features(expl)
        print('id:', k, 'coverage:', len(cov), 'Args' + str(len(expl_parsed)) + '/' + str(len(expl_parsed)) + ':', expl_parsed)