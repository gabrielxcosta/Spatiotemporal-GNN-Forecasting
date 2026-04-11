import numpy as np


def r2_score(y_true,y_pred):
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)
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
        ss_res=np.sum((yt-yp)**2)
        ss_tot=np.sum((yt-np.mean(yt))**2)
        scores.append(1-(ss_res/(ss_tot+1e-12)))
    return scores