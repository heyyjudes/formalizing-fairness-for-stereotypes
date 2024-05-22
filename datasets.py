
import pandas as pd 
import numpy as np
from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from copy import deepcopy
import acs
import utils as ut

import pickle
import fasttext
from tqdm import tqdm

class Dataset:
    def __init__(self, dataset_name: str, **kwargs) -> None:
        self.dataset_name = dataset_name
        self.x_labels = None 
        self.s_labels = None
        self.single_attr = False 
        self.acs_dict = {'income': (ACSIncome, acs.ACSIncome_categories),
                        'employment': (ACSEmployment, acs.ACSEmployment_categories),
                        'coverage': (ACSPublicCoverage, acs.ACSPublicCoverage_categories)}

        self.x, self.y, self.g = self.preprocess(dataset_name, **kwargs)

        self.split_train_test() 

    def preprocess(self, dataset_name: str, **kwargs) -> None:
        '''
        Preprocesses the dataset based on the dataset name.
        dataset_name (str): name of the dataset to preprocess
        kwargs: additional keyword arguments specific to each dataset's preprocessing method
        '''
        preprocessing_methods = {
            'income': self.preprocess_acs,
            'coverage': self.preprocess_acs,
            'employment': self.preprocess_acs,
            'bank': self.preprocess_bank,
            'lawschool': self.preprocess_lawschool,
            'biasbios': self.preprocess_biasbios
        }

        if dataset_name not in preprocessing_methods:
            raise ValueError("Dataset name not recognized")

        if dataset_name in ['income', 'coverage', 'employment']:
            return preprocessing_methods[dataset_name](dataset_name, **kwargs)
        elif dataset_name == 'bank':
            return preprocessing_methods[dataset_name](**kwargs)
        elif dataset_name == 'lawschool':
            return preprocessing_methods[dataset_name](**kwargs)
        elif dataset_name == 'biasbios':
            return preprocessing_methods[dataset_name](**kwargs)

    def get_cov_yg(self, sensitive_ind=0):
        results = []

        for i in range(self.x_train.shape[1]):
            y = self.y_train.flatten()
            if np.std(self.x_train[:, i]) > 0:
                x = (self.x_train[:, i] - np.mean(self.x_train[:, i])) / np.std(self.x_train[:, i])
                g = self.g_train[:, sensitive_ind]
                cov_y = ut.emp_cov(y, x)
                cov_g = ut.emp_cov(g, x)
                results.append({'cov(y, pg)': cov_y,
                                'cov(g, pg)': cov_g,
                                'feature': self.x_labels[i]})
        plot_df = pd.DataFrame(results)
        return plot_df


    def preprocess_biasbios(self, target_occ:str ='professor', rep:str='WE', balanced=False):
        '''
        builds train and test vectors either bag of words or word embeddings
        :param target_occ:
        :return:
        '''
        with open('BIOS.pkl', 'rb') as f:
            x = pickle.load(f)

        # extract bios
        bios = []
        for bio in x:
            bios.append({
                'gender': bio['gender'],
                'occupation': bio['title'],
                'bio': bio['bio'],
                'start_pos': bio['start_pos']
            })
        bios = pd.DataFrame(bios)

        # make class map
        self.class_map = {}
        self.reverse_class_map = {}
        self.tgt_class = target_occ
        for i, occ in enumerate(set(bios['occupation'].values)):
            self.class_map[occ] = i
            self.reverse_class_map[i] = occ

        if balanced is True:
            tgt_occ = bios[bios['occupation'] == target_occ]
            other = bios[bios['occupation'] != target_occ]
            other = other.sample(n=len(tgt_occ))
            balanced_bios = pd.concat([tgt_occ, other])
        else:
            balanced_bios = bios

        if rep == 'WE':
            print("generating fast text vectors")
            x_occ = balanced_bios.apply(lambda row: row['bio'][row['start_pos']:], axis=1).tolist()
            self.x_raw = x_occ

            ft = fasttext.load_model('cc.en.300.bin')
            x_occ, g_occ, y_all = [], [], []

            for _, row in tqdm(balanced_bios.iterrows()):
                text = row['bio'][row['start_pos']:]
                tokens = text.lower().split()
                for i, t in enumerate(tokens):
                    if i == 0:
                        vec = ft.get_word_vector(t)
                    else:
                        vec += ft.get_word_vector(t)
                if row['gender'] == 'F':
                    g_occ.append(0)
                else:
                    g_occ.append(1)
                y_all.append(self.class_map[row['occupation']])
                x_occ.append(vec)
            x_occ = np.asarray(x_occ)

        elif rep == 'BOW':
            print("generating Bag of Words")
            # Vectorized mapping to class_map
            y_all = balanced_bios['occupation'].map(self.class_map).tolist()
            # Vectorized occupation comparison 1 for target occ else 0
            y_occ = (balanced_bios['occupation'] == target_occ).tolist()
            # Vectorize gender 1 is M 0 is Female
            g_occ = np.where(balanced_bios['gender'] == 'M', 0, 1).tolist()
            # extract text starting from start poss
            x_occ = balanced_bios.apply(lambda row: row['bio'][row['start_pos']:], axis=1).tolist()
            self.x_raw = x_occ
            vectorizer = CountVectorizer(analyzer='word', stop_words='english', max_features=5000)

            x_occ = vectorizer.fit_transform(x_occ)
        else:
            raise ValueError("Must select BOW or Word Embeddings as representation")

        return x_occ, np.asarray(y_all), np.asarray(g_occ)

    def preprocess_bank(self, remove_sensitive: bool = False):
        '''
        custom preprocessing for bank dataset, one_hot encoding only
        '''
        
        self.subgroups_dict = {"cell-age0-17": [1, 0, 1, 0, 0, 0, 0], 
                                "cell-age18-29": [1, 0, 0, 1, 0, 0, 0], 
                                "cell-age30-44": [1, 0, 0, 0, 1, 0, 0],
                                "cell-age45-59": [1, 0, 0, 0, 0, 1, 0],
                                "cell-age60+": [1, 0, 0, 0, 0, 0, 1], 
                                "tele-age0-17": [0, 1, 1, 0, 0, 0, 0], 
                                "tele-age18-29": [0, 1, 0, 1, 0, 0, 0], 
                                "tele-age30-44": [0, 1, 0, 0, 1, 0, 0],
                                "tele-age45-59": [0, 1, 0, 0, 0, 1, 0],
                                "tele-age60+": [0, 1, 0, 0, 0, 0, 1]}
        self.single_attr = True
        
        df = pd.read_csv('data/bank-additional/bank-additional-full.csv', delimiter=";")
        y = df['y'].map({'no': 0, 'yes': 1})
        df = df.drop('y', axis=1)
        
        bins = [0, 18, 30, 45, 60, 120]  # Define the bin edges
        labels = ['0-17', '18-29', '30-44', '45-59', '60+']  # Define the labels for each bin
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
        
        categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 
                    'month', 'day_of_week', 'poutcome', 'age_group']
        
        df_encoded = pd.get_dummies(df, columns=categorical_cols)
        
        # get sensitive attributes
        sensitive_attr_keys = [key for key in df_encoded.keys() if (key.startswith('age_group') or key.startswith('contact'))]
        
        s = df_encoded[sensitive_attr_keys]
        x = df_encoded.drop(sensitive_attr_keys, axis=1)

        if not remove_sensitive:
            x = pd.concat((x, s), axis=1)

        self.x_labels = x.keys()
        self.s_labels = s.keys()
        
        return x.values, y.values, s.values

    def preprocess_lawschool(self, remove_sensitive: bool = False):
        self.single_attr = True
        self.exclude_attr_list = ['race1_other']
        self.subgroups_dict = {'asian-male': [1, 0, 0, 0, 0, 1],
                                'asian-feamle': [1, 0, 0, 0, 1, 0],
                                'black-male': [0, 1, 0, 0, 0, 1],
                                'black-female': [0, 1, 0, 0, 1, 0],
                                'hisp-male': [0, 0, 1, 0, 0, 1],
                                'hisp-female': [0, 0, 1, 0, 1, 0],
                                'white-male': [0, 0, 0, 1, 0, 1], 
                                'white-female': [0, 0, 0, 1, 1, 0]}
        df = pd.read_csv('data/lsac.csv', index_col=0)
        
        # get label 
        y = df['bar']
        df = df.drop('bar', axis=1)
        
        # encode categorical 
        categorical_cols = ['race1', 'gender']
        df_encoded = pd.get_dummies(df, columns=categorical_cols)
        for key in self.exclude_attr_list:
            df_encoded = df_encoded.drop(key, axis=1)
        
        sensitive_attr_keys = [key for key in df_encoded.keys() if key.startswith('race1') or key.startswith('gender')]
        s = df_encoded[sensitive_attr_keys]
        x = df_encoded.drop(sensitive_attr_keys, axis=1)
        
        if not remove_sensitive:
            x = pd.concat((x, s), axis=1)
            
        self.x_labels = x.keys()
        self.s_labels = s.keys()
        
        return x.values, y.values, s.values
        
    def preprocess_acs(self, dataset_name: str, remove_sensitive: bool = False, state='CA'):
        self.exclude_attr_list = ['RAC1P_Native Hawaiian and Other Pacific Islander alone', 
                     'RAC1P_American Indian alone', 
                     'RAC1P_Alaska Native alone', 
                     'RAC1P_Some Other Race alone', 
                     'RAC1P_Two or More Races', 
                     'RAC1P_American Indian or Alaska Native, not specified',]

        self.subgroups_dict = {"white-male": [0, 1, 0, 0, 1], 
                        "white-female": [1, 0, 0, 0, 1], 
                        "black-male": [0, 1, 1, 0, 0],
                        "black-female": [1, 0, 1, 0, 0],
                        "asian-male": [0, 1, 0, 1, 0],
                        "asia-female": [1, 0, 0, 1, 0]}

        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        data = data_source.get_data(states=[state], download=True)
        
        data_f, data_encoding = self.acs_dict[dataset_name]
        # use one hot encoding for categorical variables
        x, y, _ = data_f.df_to_pandas(data, categories=data_encoding, dummies=True)
        # remove sensitive attributes from features and labels
        for key in self.exclude_attr_list:
            if key in x.keys():
                print("removing: ", key)
                x = x.drop(key, axis=1)

        sensitive_attr_keys = [key for key in x.keys() if key.startswith("RAC1P") or key.startswith("SEX")]
        s = x[sensitive_attr_keys]

        # remove sensitive attributes from features
        x = x.drop(sensitive_attr_keys, axis=1)

        # reorder sensitive attributes
        if not remove_sensitive:
            # concatenate so sensitive attributes is at the end
            x = pd.concat((x, s), axis=1)

        # save labels
        self.x_labels = x.keys()
        self.s_labels = s.keys()
        print(f"{dataset_name} x shape: {x.shape}")
        # convert back to numpy
        x = x.dropna()
        return x.values, y.iloc[x.index].values, s.iloc[x.index].values

    def split_train_test(self, test_size=0.3, random_state=0):
        # split into train and non_train
        self.x_train, self.x_test, self.y_train, self.y_test, self.g_train, self.g_test = \
            train_test_split(self.x, self.y, self.g, test_size=test_size, random_state=random_state)
        
        self.x_valid, self.x_test, self.y_valid, self.y_test, self.g_valid, self.g_test = \
            train_test_split(self.x_test, self.y_test, self.g_test, test_size=0.5, random_state=random_state)

        if self.dataset_name == 'biasbios':
            self.y_all_train = deepcopy(self.y_train)
            self.y_all_test = deepcopy(self.y_test)
            self.y_train = np.where(self.y_train == self.class_map[self.tgt_class], 1, 0)
            self.y_test = np.where(self.y_test == self.class_map[self.tgt_class], 1, 0)
        return 

