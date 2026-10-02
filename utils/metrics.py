import numpy as np


def r2_score(y_true,y_pred):
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)
    observed=np.isfinite(y_true)
    y_true,y_pred=y_true[observed],y_pred[observed]
    if not y_true.size:
        return float("nan")
    ss_res=np.sum((y_true-y_pred)**2)
    ss_tot=np.sum((y_true-np.mean(y_true))**2)
    return 1-(ss_res/(ss_tot+1e-12))


def r2_per_horizon(y_true,y_pred):
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)
    H=y_true.shape[1]
    scores=[]
    for h in range(H):
        yt=y_true[:,h,:].reshape(-1)
        yp=y_pred[:,h,:].reshape(-1)
        scores.append(r2_score(yt,yp))
    return scores