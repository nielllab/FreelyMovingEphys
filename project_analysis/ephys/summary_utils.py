"""
summary_utils.py
"""
import sys
sys.path.insert(0, '/home/niell_lab/Documents/github/FreelyMovingEphys/')
from utils.population import Population

savepath = '/home/niell_lab/data/freely_moving_ephys/batch_files/111521/'
metadata_path = '/home/niell_lab/data/freely_moving_ephys/batch_files/111521/completed_V1_experiments.csv'
pop = Population(savepath, metadata_path)

pop.process()