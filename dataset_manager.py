import numpy as np
from sklearn.preprocessing import OneHotEncoder
import sklearn.ensemble
import pandas as pd
from typing import OrderedDict
from itertools import combinations

class DatasetManager(object):

    def __init__(self, dataset, classifier=None, nb_rows=None) -> None:
        """
        dataset: Format from anchor utils
        nb_rows: number of rows to use from dataset for training
        """
        if nb_rows is not None:
            self.dataset = dataset.train[:nb_rows]
        else:
            self.dataset = dataset.train
        
        self.nb_rows = len(self.dataset)

        if classifier is None:
            self.simple_classifier(dataset, nb_rows)
        else:
            self.classifier = classifier

        self.categorical_features = dataset.categorical_features
        self.categorical_names = dataset.categorical_names
        self.feature_names = dataset.feature_names

        self.predictions = self.predict_y()
        self.nb_classes = len(np.unique(np.array(self.predictions)))
        assert self.nb_classes == 2

        # One Hot encodes sparse matrix for dataset
        self.OH_encode()
        # Creates a dictionaries for feature values / columns
        self.preprocess_structures()
        #print(self.ibyfv)
        #print('vpf', self.values_p_feature)
        #print('fpv', self.feature_p_value)
        #print('cat_names', self.categorical_names)
        #print('fv_names', self.feature_value_names)
        

    def simple_classifier(self, dataset, nb_rows):
        c = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
        c.fit(dataset.train[:nb_rows], dataset.labels_train[:nb_rows])
        print('Train', sklearn.metrics.accuracy_score(dataset.labels_train[:nb_rows], c.predict(dataset.train[:nb_rows])))
        print('Test', sklearn.metrics.accuracy_score(dataset.labels_test, c.predict(dataset.test)))
        self.classifier = c


    def get_classifier(self):
        return self.classifier
    

    def use_synth_dataset(self, dataset, nb_rows=None):
        """
        Allows to use a raw dataset (not from anchor utils) to feed the explainer
        """
        if nb_rows is None:
            self.nb_rows = len(self.dataset)
        else:
            self.nb_rows = nb_rows
        #self.dataset = [self.encode(i) for i in dataset[:nb_rows]]
        self.dataset = dataset[:nb_rows]
        self.predictions = self.predict_y()
        self.OH_encode()
        self.preprocess_structures()


    def OH_encode(self):
        transformed_data = []
        for r in self.dataset:
            transformed_data.append([self.categorical_names[i][int(r_)] for i, r_ in enumerate(r)])
        oh_enc = OneHotEncoder(categories=self.categorical_names, handle_unknown='ignore', sparse=True)
        self.X = oh_enc.fit_transform(pd.DataFrame(transformed_data, columns=self.feature_names)).todok()
        self.feature_value_names = oh_enc.get_feature_names_out(self.feature_names)
        

    def preprocess_structures(self):
        # values in fit only are concerned NOOOO  
        self.ibyfv = dict() # instances per feature value
        for i, col in enumerate(self.X.transpose().toarray()):
            self.ibyfv.update({i: set(np.where(col)[0])})
        
        self.values_p_feature = {}  # values per feature
        self.feature_p_value = {}   # feature per feature value

        for col in self.feature_names:
            self.values_p_feature[col] = set()
            for i, fv in enumerate(self.feature_value_names):
                if col in fv:
                    self.values_p_feature[col].add(i)
                    self.feature_p_value[i] = col


    def parse_features(self, explanation):
        parsed_expl = set()
        for arg in explanation:
            parsed_expl.add(frozenset([self.feature_names[k] for k in arg]))
        return parsed_expl
    

    def predict_y(self, X=None):
        # X is a list of raw instances (as arrays of integers but not oh-encoded)
        if X is None:
            return self.classifier.predict(self.dataset)
        return self.classifier.predict(X)


    def space_size(self):
        # returns the number of possible instances
        return np.product([len(values) for values in self.values_p_feature.values()])


    def get_nb_rows(self):
        return self.nb_rows
    
    
    def generate_full_dataset(self):
        # TODO: make yield version
        instance = OrderedDict.fromkeys(self.feature_names)
        full_dataset = []
        def gen_rec(instance, features, f_i, full_dataset):
            inst_ = instance.copy()
            if f_i == len(features) - 1:
                for f in self.values_p_feature[features[f_i]]:
                    inst_[features[f_i]] = self.feature_value_names[f][len(features[f_i])+1:]
                    full_dataset.append(list(inst_.values()))
            else:
                inst_ = instance.copy()
                for f in self.values_p_feature[features[f_i]]:
                    inst_[features[f_i]] = self.feature_value_names[f][len(features[f_i])+1:]
                    gen_rec(inst_, features, f_i + 1, full_dataset)

        gen_rec(instance, list(instance.keys()), 0, full_dataset)
        return full_dataset
    

    def generate_full_neighborhood(self, origin, depth):
        #instance = OrderedDict.fromkeys(explainer.dataset.columns)
        instance = origin.copy()
        nbh_list = []

        def gen_rec(cols_, nbh_list):
            nbhs = [] 
            for nbh in nbh_list:
                for f in self.values_p_feature[cols_[0]]:
                    # create a new instance with each value for the column
                    current_value = self.feature_value_names[f][len(cols_[0])+1:]
                    if current_value != origin[cols_[0]]:
                        inst_ = nbh.copy()
                        inst_[cols_[0]] = current_value
                        nbhs.append(inst_)
            if len(cols_) > 1:
                nbhs = gen_rec(cols_[1:], nbhs)
            return nbhs
                    
        for cols_ in combinations(list(instance.keys()), depth):
            nbh_list += gen_rec(cols_, [instance])
        nlist = []
        for nbh in nbh_list:
            nlist.append(list(nbh.values()))
        return nlist

    # def instance2encoded(self, instance):
    #     encoded = []
    #     for col in self.categorical_features:
    #         encoded.append(self.categorical_names[col].index(instance[col]))
    #     return np.array(encoded)
    
    def encode(self, instance):
        """
        Encodes an instance into a vector of integers for the prediction function.
        """
        encoded = []
        for col in self.categorical_features:
            encoded.append(self.categorical_names[col].index(instance[col]))
        return np.array(encoded)
    

    def generate_neighborhoods(self, origins, max_distance):
        """
        Generates the neighborhoods of a list of instances (origins) for a given distance
        """
        print(origins[0])
        dims = tuple([len(self.categorical_names[f]) for f in self.categorical_features])
        indices = np.indices(dims).reshape(len(dims), -1).T
        distances = [np.sum(np.not_equal(indices, o), axis=1) for o in origins]
        min_distances = np.amin(distances , axis=0)
        return indices[min_distances <= max_distance]
        
