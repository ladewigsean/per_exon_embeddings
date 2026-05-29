#!/usr/bin/env python3
"""
Fetch serine protease family members from UniProt, keyed on the MEROPS/Pfam
classification rather than gene-name wildcards (which don't exist for this
super-group the way they do for CYP).

For each family in the config it builds one UniProt query of the form

    (xref:pfam-PF00089) AND (reviewed:true) AND (keyword:KW-1185) AND (taxonomy_id:33208)
     ^ family handle        ^ Swiss-Prot         ^ reference proteome     ^ clade

and streams the matches via the UniProt REST API (cursor pagination). Output per
family: a FASTA, a UniProt-accession list, and a RefSeq-protein-accession list.
The RefSeq list is the thing that feeds Sean's existing NCBI `datasets` step,
which is where exon structure (the per-exon signal) actually comes from --
UniProt gives proteins, not gene models.

Pure stdlib (urllib), no third-party deps. Read-only GETs against a public API.

Examples
--------
  # Verify query construction without hitting the network:
  python3 fetch_serine_proteases.py --print-queries

  # Small live test (10 per family) into ./test_out:
  python3 fetch_serine_proteases.py --out-dir test_out --max-per-family 10

  # Full pull, Metazoa, reviewed + reference proteomes only (defaults):
  python3 fetch_serine_proteases.py --out-dir output

  # One family, all of Eukaryota, include unreviewed:
  python3 fetch_serine_proteases.py --families S1 --taxon 2759 --include-unreviewed
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

API = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = [
    "accession", "id", "protein_name", "gene_primary", "organism_name",
    "organism_id", "length", "protein_existence", "xref_merops",
    "xref_refseq", "xref_pfam", "sequence",
]
PAGE_SIZE = 500
REQUEST_TIMEOUT = 120
RETRIES = 4
USER_AGENT = "serine-protease-fetch/1.0 (BA thesis dataset build)"


def log(msg, fh=None):
    """Print to stderr and, if given, append to the run-log file handle."""
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
    print(line, file=sys.stderr)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def family_handle(fam):
    """Return (query_fragment, kind) for the family's classification handle."""
    if fam.get("query"):
        return f"({fam['query']})", "query"
    if fam.get("pfam"):
        return f"(xref:pfam-{fam['pfam']})", "pfam"
    if fam.get("interpro"):
        return f"(xref:interpro-{fam['interpro']})", "interpro"
    if fam.get("merops"):  # list of full MEROPS ids, e.g. ["S01.001", ...]
        ids = " OR ".join(f"xref:merops-{m}" for m in fam["merops"])
        return f"({ids})", "merops"
    return None, None


def build_query(fam, taxon, reviewed_only, reference_proteome_only):
    handle, kind = family_handle(fam)
    if handle is None:
        return None, None
    parts = [handle]
    if reviewed_only:
        parts.append("(reviewed:true)")
    if reference_proteome_only:
        parts.append("(keyword:KW-1185)")  # KW-1185 = Reference proteome
    if taxon:
        parts.append(f"(taxonomy_id:{taxon})")
    return " AND ".join(parts), kind


def page_url(query, cursor=None, size=PAGE_SIZE):
    params = {
        "query": query,
        "fields": ",".join(FIELDS),
        "format": "tsv",
        "size": str(size),
    }
    if cursor:
        params["cursor"] = cursor
    return API + "?" + urllib.parse.urlencode(params)


def next_cursor(link_header):
    """Pull the cursor out of a UniProt 'Link: <...cursor=XXX...>; rel="next"' header."""
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' in part:
            url = part[part.find("<") + 1:part.find(">")]
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            return qs.get("cursor", [None])[0]
    return None


def fetch_one_page(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
                return body, resp.headers.get("Link")
        except Exception as e:  # noqa: BLE001 - retry on any transient network/HTTP error
            last_err = e
            wait = 2 ** attempt
            log(f"  request failed (attempt {attempt}/{RETRIES}): {e} -- retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"giving up after {RETRIES} attempts: {last_err}")


def fetch_family(query, max_rows, fh):
    """Yield dict rows for a family query, following cursor pagination."""
    header = None
    cursor = None
    fetched = 0
    while True:
        url = page_url(query, cursor)
        body, link = fetch_one_page(url)
        lines = body.splitlines()
        if not lines:
            break
        if header is None:
            header = lines[0].split("\t")
        rows = lines[1:]  # every page repeats the header row
        if not rows:
            break
        for raw in rows:
            yield dict(zip(header, raw.split("\t")))
            fetched += 1
            if max_rows and fetched >= max_rows:
                return
        cursor = next_cursor(link)
        if not cursor:
            break
        time.sleep(0.2)  # be polite


def first_token(value):
    """xref fields come back as 'NP_000123.1;NP_000124.1;' -> first id, no trailing dot."""
    if not value:
        return ""
    return value.replace(";", " ").split()[0] if value.strip(" ;") else ""


def write_outputs(label, prefix, rows, out_dir, raw_tsv_fh):
    fasta_path = os.path.join(out_dir, f"{prefix}_{label}.fasta")
    acc_path = os.path.join(out_dir, f"{prefix}_{label}_acc.txt")
    refseq_path = os.path.join(out_dir, f"{prefix}_{label}_refseq.txt")
    refseq_seen = set()
    n_refseq = 0
    with open(fasta_path, "w") as fa, open(acc_path, "w") as acc, open(refseq_path, "w") as rs:
        for r in rows:
            uacc = r.get("Entry", "") or r.get("accession", "")
            seq = r.get("Sequence", "")
            gene = r.get("Gene Names (primary)", "")
            org = r.get("Organism", "")
            refseq = first_token(r.get("RefSeq", ""))
            merops = r.get("MEROPS", "")
            fa.write(f">{uacc} family={label} refseq={refseq} gene={gene} OS={org}\n{seq}\n")
            acc.write(uacc + "\n")
            if refseq and refseq not in refseq_seen:
                refseq_seen.add(refseq)
                rs.write(refseq + "\n")
                n_refseq += 1
            # combined raw table (one row per protein, all families)
            raw_tsv_fh.write("\t".join([
                label, uacc, merops, refseq, gene, org,
                r.get("Length", ""), r.get("Protein existence", ""),
            ]) + "\n")
    return len(rows), n_refseq, fasta_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="serine_protease_input.json")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--taxon", default="33208", help="NCBI taxId filter; 33208=Metazoa, 2759=Eukaryota, '' to disable")
    ap.add_argument("--include-unreviewed", action="store_true", help="include TrEMBL (default: Swiss-Prot only)")
    ap.add_argument("--no-reference-proteome", action="store_true", help="drop the reference-proteome filter")
    ap.add_argument("--max-per-family", type=int, default=0, help="cap rows per family (0 = all); useful for testing")
    ap.add_argument("--families", nargs="+", help="only these family labels (e.g. S1 S8)")
    ap.add_argument("--print-queries", action="store_true", help="print constructed queries and exit (no network)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    prefix = cfg.get("prefix", "SP")
    families = cfg["families"]
    if args.families:
        wanted = set(args.families)
        families = [fam for fam in families if fam["label"] in wanted]

    reviewed_only = not args.include_unreviewed
    rp_only = not args.no_reference_proteome
    taxon = args.taxon or None

    # Build queries first; surface families with no usable handle up front.
    planned, skipped = [], []
    for fam in families:
        query, kind = build_query(fam, taxon, reviewed_only, rp_only)
        if query is None:
            skipped.append(fam["label"])
        else:
            planned.append((fam, query, kind))

    if args.print_queries:
        for fam, query, kind in planned:
            print(f"{fam['label']:>4} ({kind}): {query}")
        if skipped:
            print(f"\nSKIPPED (no pfam/interpro/merops/query handle -- fill in config): {', '.join(skipped)}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_log_path = os.path.join(args.out_dir, "logs", f"run_{ts}.log")
    os.makedirs(os.path.dirname(run_log_path), exist_ok=True)
    raw_tsv_path = os.path.join(args.out_dir, "serine_proteases_raw.tsv")

    with open(run_log_path, "w") as lf, open(raw_tsv_path, "w") as raw:
        raw.write("\t".join([
            "family", "uniprot_acc", "merops", "refseq_protein",
            "gene", "organism", "length", "protein_existence",
        ]) + "\n")
        log(f"config={args.config} taxon={taxon} reviewed_only={reviewed_only} "
            f"reference_proteome_only={rp_only} max_per_family={args.max_per_family or 'all'}", lf)
        if skipped:
            log(f"SKIPPED families (no handle, fill in config): {', '.join(skipped)}", lf)

        summary = []
        for fam, query, kind in planned:
            label = fam["label"]
            if fam.get("verify"):
                log(f"NOTE: {label} flagged verify=true but a handle was supplied -- double-check the mapping", lf)
            log(f"=== {label} [{kind}] {fam.get('name', '')} ===", lf)
            log(f"  query: {query}", lf)
            try:
                rows = list(fetch_family(query, args.max_per_family, lf))
            except Exception as e:  # noqa: BLE001
                log(f"  ERROR fetching {label}: {e}", lf)
                summary.append((label, "ERROR", "-"))
                continue
            n, n_refseq, fasta_path = write_outputs(label, prefix, rows, args.out_dir, raw)
            log(f"  {n} proteins, {n_refseq} with RefSeq xref -> {os.path.basename(fasta_path)}", lf)
            summary.append((label, n, n_refseq))
            time.sleep(0.3)

        log("=== SUMMARY (family / proteins / with-RefSeq) ===", lf)
        for label, n, n_refseq in summary:
            log(f"  {label:>4}: {n} / {n_refseq}", lf)
        log(f"raw table: {raw_tsv_path}", lf)
        log(f"run log:   {run_log_path}", lf)
        print(f"\nDone. Outputs in {args.out_dir}/ ; run log {run_log_path}")


if __name__ == "__main__":
    main()
