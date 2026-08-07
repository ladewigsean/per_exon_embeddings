"""
    Source: 
    https://github.com/UCLOrengoGroup/cath-todo/blob/master/scripts/cath-to-uniprot
    no package so just copied and altered source code
    
""" 

import csv
import logging
import os.path
import sys

import requests
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig()
LOG = logging.getLogger(__name__)

class PdbNotFoundError(Exception):
    def __init__(self, pdb_id):
        self.pdb_id = pdb_id
    
    def __str__(self):
        return "Failed to find PDB '{}' (probably obsolete)".format(self.pdb_id)

class Uniprot(object):
    def __init__(self, *, accession, gene_id, taxon_id, taxon_lineage):
        self.accession = accession
        self.gene_id = gene_id
        self.taxon_id = taxon_id
        self.taxon_lineage = taxon_lineage

    @classmethod
    def new_from_api_data(cls, data):
        accession = data['accession']
        gene_id = data['id']
        org = data['organism']
        taxon_id = org['taxonomy']
        taxon_lineage = org['lineage']
        return cls(accession=accession, gene_id=gene_id, taxon_id=taxon_id, 
            taxon_lineage=taxon_lineage)

UNIPROT_CACHE={}


UNIPROT_ACC_FOR_PDB_CACHE={}
def get_uniprot_acc_for_pdbchain(pdb_id, chain_id):

    cache_key = '{}_{}'.format(pdb_id, chain_id)
    
    pdbe_base = 'https://www.ebi.ac.uk/pdbe/api/mappings/uniprot'
    headers = {"Accept": "application/json"}

    if cache_key in UNIPROT_ACC_FOR_PDB_CACHE:
        uniprot_acc = UNIPROT_ACC_FOR_PDB_CACHE[cache_key]
    else:
        url = "{}/{}".format(pdbe_base, pdb_id)
        r = requests.get(url, headers=headers)
        if r.status_code == 404:
            raise PdbNotFoundError(pdb_id)

        LOG.debug("url: {}".format(url))
        LOG.debug("r: {}".format(r.content[0:100]))

        if not r.ok:
            r.raise_for_status()
        
        body = r.json()
        if pdb_id not in body:
            raise Exception("Error: failed to find pdb_id '{}' in body:\n{}".format(pdb_id, body))
        
        uniprot_acc = None
        for acc, entry in body[pdb_id]['UniProt'].items():
            for m in entry['mappings']:
                if m['chain_id'] is chain_id:
                    uniprot_acc = acc
                    break

        if not uniprot_acc:
            raise Exception("Error: failed to find chain '{}' in uniprot mappings for PDB id ''".format(chain_id, pdb_id))

        UNIPROT_ACC_FOR_PDB_CACHE[cache_key] = uniprot_acc

    return uniprot_acc

def cath20_id_to_uniprot(list_cath_20,warn_on_error=True):
    list_cath_20_out = []
    list_uniprot = []
    for id in list_cath_20:
        domain_id = id.split()[0]
        pdb_code = domain_id[0:4]
        chain_id = domain_id[4:5]
        uniprot_acc = None
        try:
            uniprot_acc = get_uniprot_acc_for_pdbchain(pdb_code, chain_id)
    
        except Exception as e:
            msg = "{}".format(e)
            if warn_on_error:
                print("# {} (ignoring as --warn is true)".format(e))
                continue
            else:
                raise Exception("{} (use --warn to skip)".format(e))

        if uniprot_acc:
            list_cath_20_out.append(id)
            list_uniprot.append(uniprot_acc) 
    return list_cath_20_out, list_uniprot