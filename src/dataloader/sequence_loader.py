"""
sequence_loader.py

This module contains utilities for loading protein sequences from FASTA files.
It is specifically designed to parse UniProt-style FASTA headers, such as:

>sp|A0A0C5B5G6|MOTSC_HUMAN ...
>tr|Q9XXXXX|SomeProtein ...

The parser extracts the UniProt accession (e.g., A0A0C5B5G6),
which is used throughout the CAFA6 pipeline to match IDs.
"""

from Bio import SeqIO


def parse_uniprot_fasta(fasta_path):
    """
    Parse a UniProt-style FASTA file into a dictionary:

        { uniprot_id : amino_acid_sequence }

    Example FASTA header:
        >sp|A0JP26|POTB3_HUMAN  ...
        >tr|Q91ZZ3|ABC_MOUSE    ...

    For each record:
        record.id = "sp|A0JP26|POTB3_HUMAN"
        record.id.split("|") = ["sp", "A0JP26", "POTB3_HUMAN"]
        uniprot_id = "A0JP26"

    Parameters
    ----------
    fasta_path : str
        Path to FASTA file.

    Returns
    -------
    dict
        Dictionary mapping uniprot_id → sequence string.
    """
    seq_dict = {}

    for record in SeqIO.parse(fasta_path, "fasta"):
        parts = record.id.split("|")
        if len(parts) >= 2:
            uniprot_id = parts[1]
        else:
            # fallback if unexpected header style
            uniprot_id = record.id

        seq_dict[uniprot_id] = str(record.seq)

    return seq_dict
