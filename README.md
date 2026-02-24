# Per_exon_embeddings

pe = positional Embeddings

## This repo

Essentials for classifier_runner.py. No command line arguments, changes are still being made directly in code. 

There is a lot of clutter at the moment,the important classes are 

line 343: class TransformerClassifier(nn.Module):

line 414: class MultiClassTrainer:

The whole code is still a HPO pipeline(line 614:def run_hpo_mode) because different transformer structures seem to be very hyperparameter sensitive.
Have been testing a lot of different ideas, and there might be too many different hyperparameters being tested for simultaneously.
Untested is the AlibiTransformer(not sure if I ever will) and the different schedulers(I think it might help as the model without pe tends to overfit (train: 90%, val: 82%) while the model with pe stays somewhat more equal(train: 83%, val: 78%), so maybe pe model is getting stuck in some form of local min, and some form of warm reseting scheduler could help, potentially)

All prev runs are stored online here: https://wandb.ai/per-exon-testing (I might have to invite you)
The wandb is turned off by default here in git version.
