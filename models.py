import numpy as np
import copy
import scipy.stats as stats
from scipy.optimize import minimize_scalar
from functools import partial

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from tqdm import tqdm

from sklearn.exceptions import ConvergenceWarning
import warnings
from typing import List
import utils as ut


clf_dict = {
    "LR": LogisticRegression,
    "GB": GradientBoostingClassifier,
    "XGB": xgb.XGBClassifier,
    "KNN": KNeighborsClassifier, 
    "DT": DecisionTreeClassifier,
    "NN": MLPClassifier,
    "RF": RandomForestClassifier,
}

def model_choice(clf : BaseEstimator,
                 xtrain=None, ytrain=None, scaling=True):
    param_grid_nn = {
        "mlp__alpha": [0.05, 0.1],
        "mlp__learning_rate": ["constant", "adaptive"],
        'mlp__hidden_layer_sizes': [(8, 2)] 
    }
    param_grid_knn = {
        "knn__n_neighbors": [3, 5, 7]
    }
    if scaling: 
        model = Pipeline([('scaling', StandardScaler())])
    else: 
        model = Pipeline([])
    if clf == "XBG":
        model.steps.append(("XGBoost", clf_dict[clf](objective="binary:logistic")))
    
    elif clf == "KNN": 
        temp_model = Pipeline(
            [
                ("scalar", StandardScaler()),
                ("knn", KNeighborsClassifier()),
            ]
        )
        print("running model search")
        grid_search = GridSearchCV(temp_model, param_grid_knn, n_jobs=-1, cv=5)
        
        with warnings.catch_warnings(): 
            warnings.filterwarnings("ignore",category=ConvergenceWarning) 
            grid_search.fit(xtrain, ytrain)

        # final model
        model.steps.append(("KNN", KNeighborsClassifier(grid_search.best_params_["knn__n_neighbors"])))
    elif clf == "NN":
        temp_model = Pipeline(
            [
                ("scalar", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        solver="sgd",
                        hidden_layer_sizes=(8, 2),
                        random_state=1,
                        max_iter=500,
                    ),
                ),
            ]
        )
            
        print("running model search")
        grid_search = GridSearchCV(temp_model, param_grid_nn, n_jobs=-1, cv=5)
        grid_search.fit(xtrain, ytrain)
        print(grid_search.best_params_)
        model.steps.append(("mlp", MLPClassifier(
                        solver="sgd",
                        hidden_layer_sizes=grid_search.best_params_["mlp__hidden_layer_sizes"],
                        random_state=1,
                        max_iter=500,
                        alpha=grid_search.best_params_["mlp__alpha"],
                        learning_rate=grid_search.best_params_["mlp__learning_rate"],
                    )))
        
    elif clf == "DT":
        model.steps.append(("DT", DecisionTreeClassifier(max_depth=10)))
    else:
        model.steps.append(("clf", clf_dict[clf]()))
    return model


class BoostingClassifier:
    def __init__(self, base_clf=None, coeff_arr=None, intercept_arr=None, eta=0.1):
        self.base_clf = base_clf
        self.coeff_arr = coeff_arr if coeff_arr is not None else []
        self.intercept_arr = intercept_arr if intercept_arr is not None else []
        self.eta = eta 

    def predict(self, X, thresh=0.5): 
        return self.predict_proba_1d(X) > thresh
    
    def predict_proba(self, X): 
        return np.vstack([1 - self.predict_proba_1d(X), self.predict_proba_1d(X)]).T


class EOClf(BoostingClassifier):
    def __init__(self, base_clf: BaseEstimator, pg_clf_list: List, n_bins=10):
        super().__init__(base_clf=base_clf)
        self.n_bins = n_bins
        self.bin_edges = np.linspace(0, 1, n_bins + 1)
        self.base_clf = base_clf
        self.pg_clf_list = pg_clf_list
        self.R_prime = []
        self.gamma = None

    def predict_proba_1d(self, x: np.ndarray):
        if self.a is None or self.b is None:
            raise ValueError("a and b must be set before calling predict_proba_1d, call fit")

        pg = self.pg_clf.predict_proba(x)[:, 1]
        ind = np.digitize(pg, self.bin_edges)

        R_prime_mask = np.isin(ind, self.R_prime)
        c = R_prime_mask*self.a + ~R_prime_mask*self.b # 1 if not in R', gamma if in R'
        probs = c*self.base_clf.predict_proba(x)[:, 1]
        return probs

    def check_bins_b(self, b: float, a: float, y: np.ndarray, p: np.ndarray, pg: np.ndarray):
        total_p, pos_p, neg_p = 0, 0, 0
        ind = np.digitize(pg, self.bin_edges)
        mask_y = y == 1
        marginal_mean = p[mask_y].mean()
        p_acc = []
        pg_acc = []
        for i in range(self.n_bins):
            mask_r = ind == (i + 1)
            mask_y = y == 1
            mask = mask_r & mask_y
            pg_acc.append(pg[mask])
            bin_marginal_mean = p[mask].mean()
            p_bin = (mask.sum() / mask_y.sum())
            total_p += p_bin
            if bin_marginal_mean > marginal_mean:
                p_acc.append(a * p[mask])
            else:
                p_acc.append(b * p[mask])
        pg_acc = np.concatenate(pg_acc, axis=0)
        p_acc = np.concatenate(p_acc, axis=0)
        return np.abs(ut.emp_cov(p_acc, pg_acc))

    def check_bins(self, a: float, b: float, y: np.ndarray, p: np.ndarray, pg: np.ndarray):
        total_p, pos_p, neg_p = 0, 0, 0
        ind = np.digitize(pg, self.bin_edges)
        mask_y = y == 1
        marginal_mean = p[mask_y].mean()
        p_acc = []
        pg_acc = []
        for i in range(self.n_bins):
            mask_r = ind == (i + 1)
            mask_y = y == 1
            mask = mask_r & mask_y
            pg_acc.append(pg[mask])
            bin_marginal_mean = p[mask].mean()
            p_bin = (mask.sum() / mask_y.sum())
            total_p += p_bin
            if bin_marginal_mean > marginal_mean:

                p_acc.append(a * p[mask])
            else:
                p_acc.append(b * p[mask])

        pg_acc = np.concatenate(pg_acc, axis=0)
        p_acc = np.concatenate(p_acc, axis=0)
        return np.abs(ut.emp_cov(p_acc, pg_acc))

    def fit_ls(self, x: np.ndarray, y: np.ndarray):
        pg = self.pg_clf.predict_proba(x)[:, 1]
        p = self.base_clf.predict_proba(x)[:, 1]
        ind = np.digitize(pg, self.bin_edges)
        mask_y = y == 1
        marginal_mean = p[mask_y].mean()
        pos_bins = 0
        neg_bins = 0
        total_p = 0
        for i in range(self.n_bins):
            mask_r = ind == (i+1)
            mask_y = y == 1
            mask = mask_r & mask_y
            bin_marginal_mean = p[mask].mean()
            delta = bin_marginal_mean - marginal_mean
            p_bin = (mask.sum() / mask_y.sum())
            total_p += p_bin
            # label positive bins
            if bin_marginal_mean > marginal_mean:
                self.R_prime.append(i+1)
                pos_bins += p_bin * delta * pg[mask].mean()
            else:
                neg_bins += p_bin * delta * pg[mask].mean()

        # find a, b
        if np.abs(neg_bins/pos_bins) > 1:
            partial_objective_function = partial(self.check_bins, y=y, p=p, pg=pg, a=1)
            result = minimize_scalar(partial_objective_function, bounds=(0, 1), method='Bounded')
            self.a = 1.0
            self.b = result.x
        else:
            partial_objective_function = partial(self.check_bins, y=y, p=p, pg=pg, b=1)
            result = minimize_scalar(partial_objective_function, bounds=(0, 1), method='Bounded')
            self.a = result.x
            self.b = 1.0

    def fit_exact(self, x: np.ndarray, y: np.ndarray):
        pg = self.pg_clf.predict_proba(x)[:, 1]
        p = self.base_clf.predict_proba(x)[:, 1]
        ind = np.digitize(pg, self.bin_edges)
        mask_y = y == 1
        marginal_mean = p[mask_y].mean()
        pos_bins = 0
        neg_bins = 0
        total_p = 0
        pos_ind = np.zeros((len(p)))
        neg_ind = np.zeros((len(p)))
        for i in range(self.n_bins):
            mask_r = ind == (i+1)
            mask_y = y == 1
            mask = mask_r & mask_y
            bin_marginal_mean = p[mask].mean()
            delta = bin_marginal_mean - marginal_mean
            p_bin = (mask.sum() / mask_y.sum())
            total_p += p_bin
            # label positive bins
            if bin_marginal_mean > marginal_mean:
                self.R_prime.append(i+1)
                pos_ind = np.logical_or(pos_ind, mask)
                pos_bins += p_bin * delta * pg[mask].mean()
            else:
                neg_ind = np.logical_or(neg_ind, mask)
                neg_bins += p_bin * delta * pg[mask].mean()

        p1 = p * pos_ind
        p2 = p * neg_ind
        cov1 = ut.emp_cov(p1[mask_y], pg[mask_y])
        cov2 = ut.emp_cov(p2[mask_y], pg[mask_y])
        self.a = -cov2/cov1
        print(self.a)
        self.b = 1.0



class DPMulti(BoostingClassifier):
    def __init__(self, base_clf: BaseEstimator, pg_clf_list: List):
        super().__init__(base_clf=base_clf)
        self.base_clf = base_clf
        self.pgs = pg_clf_list
        self.num_pgs = len(pg_clf_list)

    def predict_proba_1d(self, p_x: np.ndarray, pg_x: np.ndarray=None) -> np.ndarray:
        if pg_x is None:
            pg_x = p_x
        prob = self.base_clf.predict_proba(p_x)[:, 1]
        for i in range(len(self.coeff_arr)):
            pg_ind = i % self.num_pgs
            prob -= self.coeff_arr[i]*self.pgs[pg_ind].predict_proba(pg_x)[:, 1]
            prob = np.clip(prob, a_min=0, a_max=1)
        return prob

    def update(self, coeff: float):
        # only need to update the coefficient
        self.coeff_arr.append(coeff)


def fit_dp_pg_multi(x: np.ndarray,
                multi_clf: DPMulti,
                alpha:float=1e-3,
                max_iter:int=50) -> None:
    count = 0
    for _ in tqdm(range(max_iter)):
        for i in range(multi_clf.num_pgs):
            pg_ind = i % multi_clf.num_pgs
            p = multi_clf.predict_proba_1d(x)
            pg = multi_clf.pgs[pg_ind].predict_proba(x)[:, 1]
            cov = ut.emp_cov(p, pg)
            if np.abs(cov) > alpha:
                multi_clf.update(coeff=cov)
                count = 0
            else:
                count += 1
                if count == multi_clf.num_pgs:
                    return


def fit_ma_pg_multi(p_x: np.ndarray,
                multi_clf: DPMulti,
                y: np.ndarray,
                pg_x: np.ndarray = None,
                alpha:float=1e-3,
                max_iter:int=5):
    count = 0
    if pg_x is None:
        pg_x = p_x
    for _ in tqdm(range(max_iter)):
        for i in range(multi_clf.num_pgs):
            pg_ind = i % multi_clf.num_pgs
            p = multi_clf.predict_proba_1d(p_x=p_x, pg_x=pg_x)
            pg = multi_clf.pgs[pg_ind].predict_proba(pg_x)[:, 1]

            cov_ypg = ut.emp_cov(pg, y)
            curr_cov = ut.emp_cov(p, pg)
            diff = np.abs(curr_cov - cov_ypg)

            if diff > alpha:
                b = np.sign(curr_cov - cov_ypg)
                new_coeff = b*diff
                multi_clf.update(coeff=new_coeff)
                count = 0
            else:
                count += 1
                if count == multi_clf.num_pgs:
                    return


class MACov(BoostingClassifier):
    def __init__(self, base_clf, pg_clf):
        super().__init__(base_clf=base_clf)
        self.base_clf = base_clf
        self.pg_clf = pg_clf

    def predict_proba_1d(self, p_x: np.ndarray, pg_x: np.ndarray=None):
        if pg_x is None:
            pg_x = p_x
        prob = self.base_clf.predict_proba(p_x)[:, 1]

        for i in range(len(self.coeff_arr)):
            prob -= self.coeff_arr[i]*self.pg_clf.predict_proba(pg_x)[:, 1]
            prob = np.clip(prob, a_min=0, a_max=1)
        return prob

    def update(self, coeff: float):
        # only need to update the coefficient
        self.coeff_arr.append(coeff)


def fit_pg_cont(p_x: np.ndarray,
                base_clf: BoostingClassifier,
                pg_clf: BaseEstimator,
                pg_x: np.ndarray = None,
                alpha: float=1e-3,
                max_iter: int=50):
    if pg_x is None:
        pg_x = p_x
    for i in range(max_iter):
        p = base_clf.predict_proba_1d(p_x=p_x, pg_x=pg_x)
        pg = pg_clf.predict_proba(pg_x)[:, 1]
        cov = ut.emp_cov(p, pg)
        if np.abs(cov) > alpha:
            base_clf.update(coeff=cov)
        else:
            break