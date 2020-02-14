import torch
import numpy as np
import pandas as pd

from agents.dropout import DropoutAgent
from common.mushroom_env import MushroomEnv
from agents.common.mlp import MLP_Dropout

NB_STEPS = 20000
N_SEEDS = 10

for i in range(N_SEEDS):
    
    env = MushroomEnv()
    agent = DropoutAgent(env,
        MLP_Dropout,
        dropout=0.5,
        std_prior=1,
        logging=True,
        weight_decay=1e-5,
        train_freq=1,
        updates_per_train=1,
        batch_size=32,
        start_train_step=32,
        log_folder_details='Dropout',
        learning_rate=2e-3,
        verbose=True
        )

    agent.learn(NB_STEPS)