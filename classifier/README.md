# Per_exon_embeddings

pe = positional Embeddings

## This repo

Essentials for classifier_runner.py. No command line arguments, changes are still being made directly in code. 

There is a lot of clutter at the moment,the important classes are 

line 211: class PositionalEncoding(nn.Module):

line 343: class TransformerClassifier(nn.Module):

line 420: class MultiClassTrainer:

The whole code is still a HPO pipeline(line 625:def run_hpo_mode) because different transformer structures seem to be very hyperparameter sensitive.
Have been testing a lot of different ideas, and there might be too many different hyperparameters being tested for simultaneously.
currently testing just mutiplying pe by a factor, currently set as hyperparameter but might add it as learned value

All prev runs are stored online here: https://wandb.ai/per-exon-testing (I might have to invite you)

