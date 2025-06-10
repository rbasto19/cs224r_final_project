import os
import pickle
import shutil
from tqdm import tqdm

orig_data_path = "/home/ubuntu/cs224r_project/AliDiff/data/crossdocked_v1.3_rmsd1.0_pocket10"

with open("protein_filename.pkl", 'rb') as f:
    protein_filenames = pickle.load(f)

for protein_file in tqdm(protein_filenames):
    folder, protein = protein_file.split('/')
    ligand = protein.split('_pocket10')[0] + '.sdf'
    if not os.path.exists('protein_data_folder/'+folder):
        os.mkdir('protein_data_folder/'+folder)
    shutil.copy(f'{orig_data_path}/{folder}/{ligand}', f'protein_data_folder/{folder}/{ligand}')
    shutil.copy(f'{orig_data_path}/{folder}/{protein}', f'protein_data_folder/{folder}/{protein}')