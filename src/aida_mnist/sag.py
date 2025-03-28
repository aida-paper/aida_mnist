import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve
from typing import List, Union


def sag(
    u: Union[List[float], np.ndarray],
    r: Union[List[int], np.ndarray],
    k: Union[List[int], np.ndarray],
    s_des: float = 0.9,
    p_rand: float = 0.0,
    u_normalization: bool = True,
    impute: bool = True,
    n_min: int = 50,
    n_rep: int = 10,
):
    """
    Get the threshold for the uncertainty measure that achieves a desired sensitivity.
    :param u: list of uncertainties
    :param r: list of teacher rewards
    :param k: list of model update counts
    :param s_des: desired sensitivity
    :param p_rand: rate of random queries
    :param u_normalization: normalize for shifting uncertainty because of model updates
    :param impute: impute missing labels
    :param n_min: minimum number of negative samples
    :param n_rep: number of repetitions for imputation
    """
    window_len = 0
    u_array = np.array(u, dtype=float)
    r_array = np.asarray(r, dtype=int)
    k_array = np.asarray(k, dtype=int)

    window_idx = k_array >= k_array[-1] - window_len
    u_window = u_array[window_idx]
    r_window = r_array[window_idx]
    k_window = k_array[window_idx]
    known = np.logical_not(r_window == 0)

    while np.sum(r_window == -1) < n_min or np.sum(r_window == 1) < 1:
        if np.sum(window_idx) < window_len:
            # we should have at least n_min labeled samples
            return np.quantile(u, 1 - s_des)
        else:
            window_len += 1
            window_idx = k_array >= k_array[-1] - window_len
            u_window = u_array[window_idx]
            r_window = r_array[window_idx]
            k_window = k_array[window_idx]
            known = np.logical_not(r_window == 0)
    if u_normalization:
        x = k_window
        y = u_window
        uncertainty_mean = LinearRegression().fit(x.reshape(-1, 1), y)
        u_window = u_window - uncertainty_mean.predict(x.reshape(-1, 1))
        u_window += uncertainty_mean.predict(np.array([k_window[-1]]).reshape(-1, 1))

    if impute:
        failures = -r_window.copy()
        unknown = np.logical_not(known)
        y = failures[known]
        X = u_window[known].reshape(-1, 1)
        clf = LogisticRegression(penalty=None).fit(X, y)
        probas = clf.predict_proba(u_window[unknown].reshape(-1, 1))
        gammas = []
        for _ in range(n_rep):
            failures[unknown] = np.asarray(probas[:, 1] > np.random.rand(probas.shape[0]), dtype="int") * 2 - 1
            _, tpr, threshs = roc_curve(failures, u_window, pos_label=1)
            gamma = np.interp(s_des, tpr + p_rand * (1 - tpr), threshs)
            gammas.append(gamma)
        gamma = np.median(gammas)
    else:
        _, tpr, threshs = roc_curve(-r_window, u_window, pos_label=1)
        fnr = 1 - tpr
        gamma = np.interp(s_des, tpr + p_rand * fnr, threshs)
    if np.isnan(gamma):
        gamma = np.nanmin(u_window)
    return gamma
