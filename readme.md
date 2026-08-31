# Exon-aware Embeddings
This repository contains scripts for the bachelor`s thesis, "Performance of Exon-aware Embeddings in Comparison to Traditional Embeddings for General Protein Classification Tasks". 
## Provided Scripts
The following scripts were provided by a supervisor. <br />
`/dataset_creation/scripts/cluster_split.py` <br />
`/dataset_creation/scripts/identity_to_train.py` <br />
`/dataset_creation/scripts/exon_architecture.py` <br />
`/dataset_creation/scripts/report_metrics.py` <br />
`/classifier/stratified_analysis.py` <br />
## Recreate results
Download, Redundancy Reduction, and embedding are handled by `/dataset_creation/download_unfiltered.py`
### Download
```bash
python download_unfiltered.py --prefix "NCBIFGF" --output_folder data/NCBI_FGF/ --fam_input_json input_jsons/FGF_NCBI.json --download
```
The `--prefix` parameter is for file names
The `--fam_input_json` parameter is a JSON file that formats queries for the dataset; refer to existing JSONs to see the format. <br />
If querying UniProt, use the `--uniprot` flag. <br />
For Deeploc1, use `--deeploc_csv` and `--compare_fasta` parameters. <br />
### Redundancy Reduction
```bash
python download_unfiltered.py --prefix "NCBIFGF" --output_folder data/NCBI_FGF/ --fam_input_json input_jsons/FGF_NCBI.json --filter --thresh 80 --cluster_thresh 50 --min_per_class 20
```
`--thesh` is the global threshold, and `--cluster_thresh` is the cluster threshold. <br />
`--min_per_class` removes classes that have fewer than the given value total proteins after the global threshold. <br />
### Embed
```bash
python download_unfiltered.py --prefix "NCBIFGF" --output_folder data/NCBI_FGF/ --fam_input_json input_jsons/FGF_NCBI.json --embed
```
default uses  [prot t5 half precision](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc) model.
### Train Models
Embeddings should along with <prefix>.csv should be moved to `/classifier/input_data`
```bash
python classifier_runner_whole_dir.py --dir input_data/NCBI_FGF/ --project per-exon-testing --hpo_trials 20
```
`--wandb_disable` flag can be given to skip logging to a wandb project.
`--skip_per_res` flag will skip per_residue embedding type, for datasets where it is unrealistic
`--overwrite` flag will rerun exisitng results
### Graphics 
Makes bucket graphics in analysis folder
```bash
python graphics_whole_dir.py --dir output_csvs/NCBI_FGF/ --prefix_in NCBIFGF
```
### Performance
Performance Benchmark
```bash
python performance_on_dir.py -dir input_data/NCBI_FGF/ --default_avg 20
```
`--default_avg` is amount to average over

