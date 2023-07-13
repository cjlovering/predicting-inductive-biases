#%%
%load_ext autoreload
%autoreload 2

#%%

from setup import collect_probing_accuracy, organize_data, collect

# output intermediate files.
probing_accuracy = collect_probing_accuracy()
probing_accuracy

# %%

import pandas as pd 
finetune = collect('toy_1_finetune_0_lstm-toy_1.tsv')
finetune

#%%
pd.read_csv('../results/stats/toy_1_probing_weak_lstm-toy_1.tsv', sep='\t')


# %%

import os 
if not os.path.exists("./figures"):
	os.mkdir("./figures")

if not os.path.exists("./files"):
	os.mkdir("./files")

#%% 
import setup	

setup.main()
#%%
pdata = pd.read_csv('files/probing.tsv', sep='\t')
total_rel = pdata['total_mdl_weak'] / pdata['total_mdl_strong']
rel = pdata['weak-mdl'] / pdata['strong-mdl']
total_rel, rel	    # these two are the same 


# %%
# making plots across evidence 


# %%

#%%
pdata = pd.read_csv('files/probing_accuracy.tsv', sep='\t')
# pdata['total_mdl_weak']


#%%
import tables 

tables.main()

#%%

# this doesn't work because there is currently only a single feature
# so only one value of relative extractabiilty 
import lineplots

lineplots.main()


# %%


results = pd.read_table('files/results.tsv')
probing = pd.read_table("files/probing.tsv")
#%%
results
#%%
results.columns
#%%
results['weak-error']


# %%

probing

# %%
