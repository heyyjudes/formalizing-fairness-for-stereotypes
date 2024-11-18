import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator
from typing import List, Dict
from sklearn.metrics import accuracy_score

def eo_metric(y_true: np.ndarray,
              y_pred: np.ndarray,
              pg: np.ndarray):
    mask = y_true == 1
    return emp_cov(y_pred[mask], pg[mask])

def cov_corr_summary(y: np.array, p: np.array, g: np.array, pg: np.array):
    assert y.shape == p.shape
    assert pg.shape == g.shape
    assert y.shape == g.shape
    assert len(y.shape) == 1

    summary = {}
    summary['cov(p, pg)'] = emp_cov(p, pg)
    summary['cov(p, g)'] = emp_cov(p, pg)
    summary['cov(y, g)'] = emp_cov(y, g)

    summary['corr(p, pg)'], _ = stats.pearsonr(p, pg)
    summary['corr(p, g)'], _ = stats.pearsonr(p, g)
    summary['corr(y, g)'], _ = stats.pearsonr(y, g)
    return summary

def results_summary(pg_dict: Dict["str", BaseEstimator],
                    p: np.ndarray,
                    x: np.ndarray,
                    y: np.ndarray,
                    g: np.ndarray,) -> Dict:
    ''' return accuracy, acc, mertic, cov(p, g_1), cov(p, g_2) '''
    results = {}
    p_acc = accuracy_score(y_true=y, y_pred=p > 0.5)
    results['accuracy'] = p_acc
    mask = g == 1
    group1 = (p > 0.5)[mask].mean()
    group2 = (p > 0.5)[~mask].mean()
    results['DP_G'] = np.abs(group1 - group2)
    mask_y = y == 1
    group1 = (p > 0.5)[(mask & mask_y)].mean()
    group2 = (p > 0.5)[(~mask & mask_y)].mean()
    results['EO_G'] = np.abs(group1 - group2)


    for key, pg in pg_dict.items(): 
        pg_vals = pg.predict_proba(x)[:, 1]
        # pg DP
        results[f'cov(p, {key})'] = emp_cov(p, pg_vals)
        # pg EO 
        mask_y = y == 1
        results[f'cov(p, {key} |y=1)'] = emp_cov(p[mask_y], pg_vals[mask_y])
    return results


def emp_cov(x: np.array, y: np.array):
    assert x.shape == y.shape
    assert len(x.shape) == 1
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    return np.sum((x - mu_x) * (y - mu_y)) / (len(x) - 1)




