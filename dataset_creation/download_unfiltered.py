import argparse
import scripts.grab_data
import os
import scripts.filter_data
import json
import scripts.embed_pers
import scripts.train_val_test_splits
import sys

if __name__ == '__main__':
    #python download_unfiltered.py --prefix "SerProt" --output_folder data/SerineProt/ --fam_input_json serineprot.json --uniprot --download --filter --embed
    

    parser = argparse.ArgumentParser(description="Train or optimize a multi-class SCPP classifier.")
    parser.add_argument("--prefix", required=True, help="output prefix")
    parser.add_argument("--output_folder", required=True, help="output folder location")
    parser.add_argument("--fam_input_json", help="family json used to search NCBI")
    parser.add_argument("--uniprot", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--thresh",default="30", help="percent identity theshold" )
    parser.add_argument("--min_per_class",type = int, default=60, help="min amount in a class" )
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--embed_prefix_addon", help="adds addon to prefix for embed incase multiple models, include\"_\" ", default="")
    parser.add_argument("--embed_model", help="model used for embedding, not all supported", default="Rostlab/prot_t5_xl_half_uniref50-enc")
    
    args = parser.parse_args()
    no_filter_fasta =os.path.join(args.output_folder ,f"{args.prefix}_no_filter.fasta")
    no_filter_cuts_csv =  os.path.join(args.output_folder ,f"{args.prefix}_no_filter_cuts.csv")
    fasta =os.path.join(args.output_folder ,f"{args.prefix}.fasta")
    csv =  os.path.join(args.output_folder ,f"{args.prefix}.csv")
    
    if args.download:
        print("Downloading...")
        if args.fam_input_json is None:
            raise ValueError("need fam_input_json when download set ")
        with open(args.fam_input_json,"r") as f:
            main_family_dict = json.load(f)
        if args.uniprot:
            parsed_data = scripts.grab_data.download_entrez_from_fam_dict_uniprot(main_family_dict,log_file=f"{args.prefix}_download_log.txt")
        else:
            parsed_data = scripts.grab_data.download_entrez_from_fam_dict(main_family_dict)
        
        #parsed_data =download_entrez_rna(main_family_dict)
        no_filter = scripts.filter_data.flatten_dict(parsed_data)
        
        no_filter_csv = os.path.join(args.output_folder,f"{args.prefix}_no_filter.csv")
        no_filter_acc = os.path.join(args.output_folder,f"{args.prefix}_no_filter_acc.txt" )
        scripts.grab_data.save_fasta(no_filter,no_filter_fasta)
        scripts.grab_data.save_csv(no_filter,no_filter_csv)
        scripts.grab_data.save_acc(no_filter,no_filter_acc)
        scripts.grab_data.get_exon_data(no_filter_csv, no_filter_acc,no_filter_cuts_csv,args.output_folder)
    if args.filter:
        print("Filtering...")
        parsed_data = scripts.grab_data.read_metadata_fasta(no_filter_fasta,no_filter_cuts_csv)

        whisk = 3
        scripts.filter_data.filter_outliers_all_graphic(parsed_data,args.output_folder,prefix=args.prefix,whisk=whisk)
        
        combined_all_filtered = scripts.filter_data.splice_aware_filter_attmept1(parsed_data,args.prefix,args.output_folder, ident_threshold=args.thresh,mode="mmseqs",whisk=whisk,min_nummer=args.min_per_class)
        scripts.grab_data.save_fasta(combined_all_filtered,fasta)
        scripts.grab_data.save_csv_splits(combined_all_filtered,csv )
        scripts.train_val_test_splits.split_train_val_test(csv,min_per_class=args.min_per_class)
    if args.embed:
        print("Embedding...")
        scripts.embed_pers.embed(fasta,csv,args.embed_prefix_addon,embedding_types=["per_prot","per_exon","per_res","fixed_length_chunks","fixed_total_chunks"],model_name = args.embed_model)