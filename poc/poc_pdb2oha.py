import numpy as np
import pathlib
import pandas as pd

csv_path = pathlib.Path().absolute() / "samples" / "pdb_data_seq.csv"

# Let's read from the CSV containing the protein sequences first
# structureID --> PDB ID (this makes it easier to get the 3D structures of the sequences)
pdb = pd.read_csv(csv_path)
pdb # Just to show the table in Jupyter

amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def one_he(sequence, alphabet=amino_acids):
    ab_index = {aa: i for i, aa in enumerate(alphabet)}
    ab_array = np.zeros((len(sequence), len(alphabet)))

    for i, aa in enumerate(sequence):
        if aa in ab_index:
            ab_array[i, ab_index[aa]] = 1
    return ab_array

pseq_smp = pdb.sample(n=1).loc[:,"sequence"].values[0]
print(pseq_smp)
encoded_sequence = one_he(pseq_smp)
print(encoded_sequence)