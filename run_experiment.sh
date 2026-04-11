#!/bin/bash

docker run --gpus all -d \
-v /media:/media \
-w /media/work/gabrielcosta \
--name gabrielcosta_experiment_TGCN_residual \
-u 1083:1083 \
-e TMPDIR=/media/work/tmp \
-e PYTHONUNBUFFERED=1 \
gnn-env python3 experiment_TGCN_residual.py

docker run --gpus all -d \
-v /media:/media \
-v /media/work/gabrielcosta/tmp:/tmp \
-w /media/work/gabrielcosta \
--name gabrielcosta_experiment_GCLSTM_wikimaths_long \
-u 1083:1083 \
-e PYTHONUNBUFFERED=1 \
gnn-env python3 main.py

docker run --gpus all -d \
-v /media:/media \
-v /media/work/gabrielcosta/tmp:/tmp \
-w /media/work/gabrielcosta \
--name gabrielcosta_spectral_analysis \
-u 1083:1083 \
-e PYTHONUNBUFFERED=1 \
gnn-env python3 spectral_analysis.py

ps -u gabrielcosta -o pid,cmd,lstart --sort=-lstart | grep python

source /media/work/fernandoduarte/ASOP/venv/bin/activate