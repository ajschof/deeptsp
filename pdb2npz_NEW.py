import os
import gzip
import numpy as np
from Bio.PDB import PDBParser, CaPPBuilder, Polypeptide
from scipy.spatial.distance import pdist, squareform
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio import BiopythonDeprecationWarning
import gc
import pathlib
import time

warnings.filterwarnings("ignore", category=PDBConstructionWarning)
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

def one_hot(residue):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    one_hot = [0] * len(amino_acids)
    index = amino_acids.find(residue)
    if index != -1:
        one_hot[index] = 1
    return one_hot

def pdb_to_one_hot_array(structure):
    ppb = CaPPBuilder()
    one_hot_list = []
    for pp in ppb.build_peptides(structure):
        for residue in pp:
            residue_name = residue.get_resname()
            if residue_name in Polypeptide.standard_aa_names:
                one_hot_list.append(one_hot(Polypeptide.three_to_one(residue_name)))
    return np.array(one_hot_list)

def pdb_to_distance_matrix(structure):
    ppb = CaPPBuilder()
    coords = []
    for pp in ppb.build_peptides(structure):
        for residue in pp:
            residue_name = residue.get_resname()
            if residue_name in Polypeptide.standard_aa_names:
                ca_atom = residue['CA']
                coords.append(ca_atom.get_coord())
    
    if len(coords) < 2:
        print(" - Insufficient coordinates for distance matrix calculation.")
        return np.array([])
    
    distance_vector = pdist(coords)
    return squareform(distance_vector)

def is_gzip_file(filepath):
    with open(filepath, 'rb') as file:
        return file.read(2) == b'\x1f\x8b'

def pdb_to_dihedral_angles(structure):
    ppb = CaPPBuilder()
    phi_angles = []
    psi_angles = []
    for pp in ppb.build_peptides(structure):
        angles = pp.get_phi_psi_list()
        for phi, psi in angles:
            phi_angles.append(phi if phi is not None else np.nan)
            psi_angles.append(psi if psi is not None else np.nan)
    return np.array(phi_angles), np.array(psi_angles)

pdb_folder = pathlib.Path.cwd() / "pdb_database"
parser = PDBParser()
num_processed = 0

data_dict = {}

output_folder = "npz_files"
os.makedirs(output_folder, exist_ok=True)

start_time = time.time()
time_spent = 0

for file in os.listdir(pdb_folder):
    if file.endswith('.pdb.gz'):
        pdb_path = os.path.join(pdb_folder, file)
        if not is_gzip_file(pdb_path):
            print(f"Skipping non-gzipped or corrupted file: {file}")
            continue

        num_processed += 1
        
        try:
            with gzip.open(pdb_path, 'rt') as pdb_file:
                structure = parser.get_structure(file[:-7], pdb_file)

                one_hot_array = pdb_to_one_hot_array(structure)
                distance_matrix = pdb_to_distance_matrix(structure)
                phi_angles, psi_angles = pdb_to_dihedral_angles(structure)

                output_path = os.path.join(output_folder, f"{file[:-7]}.npz")
                np.savez_compressed(output_path, one_hot=one_hot_array, distance_matrix=distance_matrix, phi_angles=phi_angles, psi_angles=psi_angles)

                del one_hot_array
                del distance_matrix
                del phi_angles
                del psi_angles
        
        except EOFError:
            print(f"\nError processing file {file}: The file is corrupted or truncated.")
        except Exception as e:
            print(f"\nError processing file {file}: {e}")

        time_spent += time.time() - start_time
        start_time = time.time()

        average_time_per_file = time_spent / num_processed
        remaining_files = len(os.listdir(pdb_folder)) - num_processed
        time_left = average_time_per_file * remaining_files

        print(f"\rProcessing file: {file} | Estimated time left: {time_left / 60:.2f} minutes", end="", flush=True)