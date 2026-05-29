# Serine protease dataset builder

Fetches serine protease family members from UniProt to build a per-exon vs
per-protein benchmark, keyed on the **MEROPS / Pfam classification** rather than
gene-name wildcards.

## Why not gene names

The CYP pipeline in `dataset_creation/` works by wildcarding gene nomenclature
(`CYP2A*`) in NCBI Entrez. Serine proteases have no equivalent systematic
nomenclature (`PRSS*`, `KLK*`, `F2`, `F11`, `TMPRSS*`, …), so the reliable handle
is the classification cross-reference instead:

- The "16 superfamilies / ~40 families" breakdown of serine proteases **is the
  MEROPS classification** (clan = superfamily, family = `Sxx`).
- Each MEROPS family maps ~1:1 to a Pfam/InterPro domain (S1↔PF00089,
  S8↔PF00082, S9↔PF00326, …).
- UniProt cross-references both, queryable as `xref:pfam-PFxxxxx`.

Each family becomes one UniProt query with consistent filters:

```
(xref:pfam-PF00089) AND (reviewed:true) AND (keyword:KW-1185) AND (taxonomy_id:33208)
 ^ family handle       ^ Swiss-Prot        ^ reference proteome    ^ clade (Metazoa)
```

## Per-exon dependency

UniProt returns the **protein**, not the gene model — and the per-exon features
need exon/intron boundaries, which come from a genome annotation (NCBI `datasets`
in the existing CYP pipeline). So a protein is only usable if it also has a RefSeq
gene model. The fetcher therefore writes a per-family **RefSeq-protein-accession
list** (`SP_<fam>_refseq.txt`) — that list is the input to the `datasets` /
exon-extraction step. For a convergent-genes benchmark the gene model is where the
exon-architecture signal lives, so annotation quality matters most there.

## Files

- `serine_protease_input.json` — family table (`label` / `clan` / `name` +
  `pfam` / `interpro` handle). Populated: S1, S8, S9, S10, S28, S14, S49. Left
  blank with `"verify": true` (skipped until filled): S33, S26, S16.
- `fetch_serine_proteases.py` — stdlib-only (urllib); reads the config, queries
  UniProt REST per family with cursor pagination, and writes per-family FASTA +
  UniProt-acc list + RefSeq-acc list, a combined `serine_proteases_raw.tsv`, and a
  timestamped run log under `<out>/logs/`.

## Usage

```bash
cd serine_protease_dataset
python3 fetch_serine_proteases.py --print-queries        # sanity-check queries, no network
python3 fetch_serine_proteases.py --out-dir output       # full pull: Metazoa, reviewed + RP
```

Knobs: `--taxon 2759` (Eukaryota) · `--include-unreviewed` · `--no-reference-proteome`
· `--families S1 S8` · `--max-per-family N` (testing).

## Next steps

- [ ] Fill the three blanked families (S33, S26, S16): confirm Pfam/InterPro on the
      MEROPS family pages, or drop them.
- [ ] Spot-check the seven populated mappings — the `merops` column in
      `serine_proteases_raw.tsv` should match each row's family label
      (e.g. S1 rows should carry `S01.*`).
- [ ] Pick the clade (Metazoa vs Eukaryota) and a target N per family.
- [ ] Feed `SP_<fam>_refseq.txt` into the `datasets`/exon step to recover exon
      structure, then reuse the existing per-exon / per-protein embedding path.
- [ ] For the convergent-genes test, clan PA / family S1 is the strongest starting
      point (large, with independent recruitments of the same fold).

## Status

Query construction verified for all seven populated families; a capped live fetch
of S1 returned proteins whose MEROPS cross-references were `S01.*`, confirming the
`PF00089 → S1` handle. A full pull has not yet been run.
