import numpy as np
import gzip
import os, random
import pathlib
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio import BiopythonDeprecationWarning
import warnings

warnings.filterwarnings("ignore", category=PDBConstructionWarning)
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

def get_ca_coordinates(pdb_file):
    with gzip.open(pdb_file, 'rt') as file:
        parser = PDBParser()
        structure = parser.get_structure("protein", file)
        model = structure[0]

        ca_coordinates = []
        for chain in model:
            for residue in chain:
                if residue.has_id("CA"):
                    ca_coordinates.append(residue["CA"].coord)

    return np.array(ca_coordinates)

def distance_matrix(coords):
    return np.sqrt(((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=-1))

# pdb_choice = os.path.join(os.getcwd(), random.choice(os.listdir("PDBs")))
pdb_choice = pathlib.Path('.').absolute() / 'pdb_database' / random.choice(os.listdir("pdb_database"))

print("\n" + str(pdb_choice) + "\n")

ca_coords = get_ca_coordinates(pdb_choice)
dist_matrix = distance_matrix(ca_coords)

print(dist_matrix)