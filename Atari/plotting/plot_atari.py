import os
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import pickle
import numpy as np
import matplotlib.pyplot as plt

games = ['alien','amidar','assault','asterix']

ide_seeds = [1,2,3]
qrdqn_seeds = [1,2,3]
eqrdqn_seeds = [1]
bootstrapped_seeds = [1,2,3]
learners = ['IDS','QRDQN','Bootstrapped']

#plt.gcf().subplots_adjust(bottom=0.2)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
fig.tight_layout()
fig.subplots_adjust(left=0.05,bottom=0.15,top=0.9,hspace=0.4)

for idx,game in enumerate(games):

    game_scores_min = []
    game_scores_max = []
    game_scores_mean = []

    for learner in learners:

        if learner == 'IDS':
            seeds = ide_seeds
        elif learner == 'QRDQN':
            seeds = qrdqn_seeds
        elif learner == 'Bootstrapped':
            seeds = bootstrapped_seeds
        else:
            seeds = eqrdqn_seeds

        avg_scores = []

        learner_scores = []

        for seed in seeds:

            filename = 'results/Atari/summary/{}/{}_scores{}'.format(learner,game,str(seed))
            scores = np.array(pickle.load(open(filename, 'rb')))
            #scores = savgol_filter(scores,9,3)
            learner_scores.append(scores)
        
        game_scores_mean.append(np.stack(learner_scores).mean(axis=0))
        game_scores_min.append(np.stack(learner_scores).min(axis=0))
        game_scores_max.append(np.stack(learner_scores).max(axis=0))

    game_scores_mean = np.stack(game_scores_mean)
    game_scores_min = np.stack(game_scores_min)
    game_scores_max = np.stack(game_scores_max)

    coordinates = np.array([[0,0],[0,1],[1,0],[1,1]])


    for i in range(game_scores_mean.shape[0]):
        axs[coordinates[idx,0],coordinates[idx,1]].plot(game_scores_mean[i,:],label=learners[i])
        axs[coordinates[idx,0],coordinates[idx,1]].fill_between(np.arange(50),
               game_scores_min[i,:],
               game_scores_max[i,:],
               alpha=0.2)
    
    #coordinates[idx][0],coordinates[idx][1]


    axs[coordinates[idx,0],coordinates[idx,1]].set_title('{}'.format(game).capitalize(),fontsize = 22)
    if idx > 1:
        axs[coordinates[idx,0],coordinates[idx,1]].set_xlabel('Million Frames', fontsize = 22)
    if idx== 0 or idx==2:
        axs[coordinates[idx,0],coordinates[idx,1]].set_ylabel('Score', fontsize = 22)
    axs[coordinates[idx,0],coordinates[idx,1]].tick_params(axis="x", labelsize=22)
    axs[coordinates[idx,0],coordinates[idx,1]].set_yticks([])

    if idx ==2:
        axs[coordinates[idx,0],coordinates[idx,1]].axis([0,50,0,20000])

    if idx ==3:
        axs[coordinates[idx,0],coordinates[idx,1]].legend(fontsize = 14)
    
plt.show()

