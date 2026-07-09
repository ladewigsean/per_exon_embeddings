from pyfaidx import Fasta
def deeploc_grab(file_path):
    data = {}
    fasta = Fasta(str(file_path))
    for seq in fasta:
        split_header = seq.name.split(" ")
        if len(split_header ) !=3:
            split_header = split_header + ["train"]
        split_header = split_header + [str(seq)]
        data[split_header[0]] =split_header[1:] 
        
    
    return headers