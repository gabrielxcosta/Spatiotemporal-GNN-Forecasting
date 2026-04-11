import os
os.environ["MPLCONFIGDIR"]="./.cache_matplotlib"
os.environ["TMPDIR"]="/tmp"
os.environ["TEMP"]="/tmp"
os.environ["TMP"]="/tmp"

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import math
import numpy as np
import torch
import torch.optim as optim

from itertools import product

#from loaders.englandcovid_loader import EnglandCovidDatasetLoaderLocal
#from loaders.chickenpox_loader import ChickenpoxDatasetLoaderLocal
from loaders.wikimaths_loader import WikiMathsDatasetLoaderLocal
#from loaders.montevideobus_loader import MontevideoBusDatasetLoaderLocal
#from loaders.twittertennis_loader import TwitterTennisDatasetLoaderLocal

from utils.utilities import set_seed,build_config_name,adjust_temporal_params
from utils.metrics import r2_score,r2_per_horizon
from utils.plotting import plot_loss,plot_regression,plot_temporal
from utils.dataloaders import build_adjacency,build_dataloaders
from utils.training import train_epoch,evaluate

#from models.convolutional.aagcn import AAGCN_Model
#from models.convolutional.slcnn import SLCNN
#from models.attention.cast import CaST
#from models.attention.stgraformer import STGraphormer
#from models.attention.staeformer import STAEformer
#from models.attention.tgat import TGAT
#from models.recurrent.gconvlstm import G_Conv_LSTM
#from models.recurrent.tgcn import T_GCN
#from models.recurrent.dcrnn import G_DCRNN
#from models.attention.stgnn import STTransformerSpectralPE
#from models.recurrent.egcno import Evolve_GCN_O
#from models.attention.gman import GMAN
#from models.convolutional.lsgcn import LSGCN
#from models.convolutional.stgcn import STGCN
#from models.convolutional.graph_wavenet import GraphWaveNet
#from models.convolutional.mtgnn import MTGNNModel
#from models.recurrent.gclstm import GC_LSTM
#from models.recurrent.mpnnlstm import MPNN_LSTM 
#from models.recurrent.dygrae import DyGrAE
#from models.convolutional.lsgcn import LSGCN

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ARCHITECTURE="EvolveGCNO"
RESULTS_ROOT="results_wikimaths_long"
os.makedirs(RESULTS_ROOT,exist_ok=True)


def run(seed,cfg):

    start=time.time()

    set_seed(seed)

    config_name=build_config_name(cfg)

    results_dir=os.path.join(
        RESULTS_ROOT,
        ARCHITECTURE,
        config_name,
        f"seed_{seed}"
    )

    os.makedirs(results_dir,exist_ok=True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #loader=EnglandCovidDatasetLoaderLocal()
    loader=WikiMathsDatasetLoaderLocal()
    #loader=MontevideoBusDatasetLoaderLocal()
    #loader = ChickenpoxDatasetLoaderLocal()
    #dataset = list(loader.get_dataset())
    dataset=list(loader.get_dataset(lags=cfg["lags"]))

    first=dataset[0]

    edge_index = first.edge_index.clone().detach().long()
    edge_weight=np.ones(edge_index.shape[1]) if first.edge_weight is None else first.edge_weight
    edge_weight=torch.tensor(edge_weight,dtype=torch.float32)

    #A = build_adjacency(edge_index,edge_weight,first.num_nodes)

    X=[s.x for s in dataset]
    X=np.stack(X)

    if X.ndim==2:
        data=X[...,None]
    else:
        data=X

    print("\n=== DATA DEBUG ===")
    print("data shape:", data.shape)
    print("data min/max:", data.min(), data.max())
    print("data mean/std:", data.mean(), data.std())
    print("has NaN:", np.isnan(data).any())
    print("has Inf:", np.isinf(data).any())

    N=data.shape[1]

    tr,va,te,lags,horizon=build_dataloaders(
        data,
        cfg["lags"],
        cfg["horizon"],
        cfg["batch_size"]
    )

    #model = Evolve_GCN_H(
    #    in_ch=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"],
    #    num_nodes=N
    #).to(device)

    #model = LSGCN(
    #    num_nodes=N,
    #    in_ch=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"],
    #    edge_drop=cfg["edge_drop"]
    #).to(device)

    #model = DyGrAE(
    #    in_ch=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"]
    #).to(device)

    #model = G_DCRNN(
    #    in_ch=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"],
    #).to(device)
    
    #model = G_Conv_LSTM(
    #    in_ch=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"],
    #).to(device)
    
    #model = AAGCN_Model(
    #    num_nodes=N,
    #    in_ch=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"],
    #    edge_drop=cfg["edge_drop"]
    #).to(device)

    #model = DyGrAE(
    #    in_ch=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"]
    #).to(device)

    #model = STGraphormer(
    #    num_nodes=N,
    #    in_dim=data.shape[-1],
    #    d_model=cfg["hidden"],
    #    heads=4,
    #    layers=1,
    #    horizon=cfg["horizon"],
    #).to(device)
    
    #model = STAEformer(
    #    num_nodes=N,
    #    in_steps=cfg["lags"],
    #    out_steps=cfg["horizon"],
    #    input_dim=data.shape[-1],
    #    output_dim=1,
    #    input_embedding_dim=cfg["hidden"],
    #    spatial_embedding_dim=cfg["hidden"],
    #    adaptive_embedding_dim=cfg["hidden"],
    #    feed_forward_dim=cfg["hidden"]*4,
    #    num_heads=4,
    #    num_layers=1,
    #    dropout=cfg["dropout"],
    #    edge_drop=cfg["edge_drop"]
    #).to(device)

    #model = SLCNN(
    #    num_nodes=N,
    #    in_ch=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"],
    #    edge_drop=cfg["edge_drop"]
    #).to(device)

    #model = TGAT(
    #    in_dim=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    time_dim=cfg["hidden"],
    #).to(device)

    #model = STTransformerSpectralPE(
    #    num_nodes=N,
    #    in_ch=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"],
    #    edge_drop=cfg["edge_drop"],
    #).to(device)

    model = Evolve_GCN_O(
        in_ch=data.shape[-1],
        hidden=cfg["hidden"],
        horizon=cfg["horizon"],
        dropout=cfg["dropout"]
        #num_nodes=N
    ).to(device)

    #model = GMAN(
    #    num_nodes=N,
    #    in_dim=data.shape[-1],
    #    d_model=cfg["hidden"],
    #    out_dim=1,
    #    heads=4,
    #    num_layers=1,
    #    horizon=cfg["horizon"],
    #    spatial_mode="attention"
    #).to(device)

    #model = STGCN(
    #    num_nodes=N,
    #    in_ch=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"]
    #).to(device)

    #model = GraphWaveNet(
    #    adj=A,
    #    num_nodes=N,
    #    input_dim=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"]
    #).to(device)

    #model = MTGNNModel(
    #    num_nodes=N,
    #    in_ch=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"],
    #    seq_length=cfg["lags"],
    #    edge_drop=cfg["edge_drop"]
    #).to(device)

    #model = GC_LSTM(
    #    in_ch=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"]
    #).to(device)
    
    #model = MPNN_LSTM(
    #    in_ch=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"],
    #    lags=cfg["lags"],
    #    num_nodes=N
    #).to(device)

    #model = T_GCN(
    #    in_ch=data.shape[-1],
    #    hidden=cfg["hidden"],
    #    horizon=cfg["horizon"],
    #    dropout=cfg["dropout"]
    #).to(device)

    #model = CaST(
    #    num_nodes=N,
    #    input_dim=data.shape[-1],
    #    hidden_dim=cfg["hidden"],
    #    seq_len=cfg["lags"],
    #    horizon=cfg["horizon"]
    #).to(device)

    optimizer=optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=1e-4
    )

    warmup_epochs=cfg["warmup"]
    cosine_epochs=max(1,cfg["epochs"]-warmup_epochs)

    cosine_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs
    )

    def warmup_lambda(epoch):
        if epoch<warmup_epochs:
            return float(epoch+1)/float(max(1,warmup_epochs))
        return 1.0

    warmup_scheduler=torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=warmup_lambda
    )

    best=float("inf")
    patience_ctr=0
    patience=cfg["patience"]

    train_losses=[]
    val_losses=[]

    for epoch in range(cfg["epochs"]):
        tr_loss, did_step = train_epoch(model,tr,optimizer,device,edge_index,edge_weight)
        val_loss,_,_=evaluate(model,va,device,edge_index,edge_weight)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{cfg['epochs']} train={tr_loss:.6f} val={val_loss:.6f}")

        if did_step:
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

        if val_loss < best:
            best = val_loss
            best_state = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    model.load_state_dict(best_state)

    test_mse,pred,true=evaluate(model,te,device,edge_index,edge_weight)

    y_true=true.reshape(-1)
    y_pred=pred.reshape(-1)

    r2_global=r2_score(y_true,y_pred)
    r2_horizons=r2_per_horizon(true,pred)
    r2=float(np.mean(r2_horizons))

    rmse=float(np.sqrt(test_mse))
    mae=float(np.mean(np.abs(y_true-y_pred)))
    mape=float(np.mean(np.abs((y_true-y_pred)/(y_true+1e-8)))*100)

    plot_loss(train_losses,val_losses,config_name,results_dir)
    plot_regression(y_true,y_pred,config_name,results_dir)
    plot_temporal(true.mean(axis=(1,2)),pred.mean(axis=(1,2)),config_name,results_dir)

    runtime=time.time()-start

    metrics={
        "seed":seed,
        "test_mse":float(test_mse),
        "test_rmse":rmse,
        "test_mae":mae,
        "test_mape":mape,
        "test_r2_global":float(r2_global),
        "test_r2_mean_horizon":float(r2),
        "test_r2_per_horizon":[float(v) for v in r2_horizons],
        "runtime_sec":runtime,
        "epochs_ran":len(train_losses),
        "config":cfg
    }

    with open(os.path.join(results_dir,"metrics.json"),"w") as f:
        json.dump(metrics,f,indent=4)

    return test_mse


def experiment():

    grid={
        "lags":[50],
        "hidden":[32,64],
        "lr":[1e-3],
        "batch_size":[32],
        "dropout":[0.2],
        "edge_drop":[0.1],
        "horizon":[20],
        "epochs":[200],
        "warmup":[5],
        "patience":[20]
    }

    configs=[dict(zip(grid.keys(),v)) for v in product(*grid.values())]

    for cfg in configs:

        print("\nCONFIG:",cfg)

        scores=[]

        for seed in range(10):

            print("RUN seed=",seed)

            mse=run(seed,cfg)

            scores.append(mse)

        mean=np.mean(scores)
        std=np.std(scores)

        print("FINAL RESULT",mean,"±",std)


def main():
    experiment()


if __name__=="__main__":
    main()