import subprocess
import pickle
import shlex
from tqdm import tqdm
import sys
sys.path.append('/home/ubuntu/cs224r_project/AliDiff/')
import datasets_alidiff

with open('/home/ubuntu/cs224r_project/AliDiff/data/dpo_train_set.pkl', 'rb') as f:
    dpo_train_set = pickle.load(f)

data_path = '/home/ubuntu/cs224r_project/AliDiff/data/crossdocked_v1.3_rmsd1.0_pocket10'
for i in tqdm(range(len(dpo_train_set))):
    pl_pair = dpo_train_set[i]
    prot_filename = pl_pair.protein_filename
    win_lig_filename = pl_pair.ligand_filename
    # lose_lig_filename = pl_pair.ligand_filename2

    subprocess.run(shlex.split(f"python encoder/encode.py --pdb_file {data_path}/{prot_filename} --sdf_file {data_path}/{win_lig_filename} --output_name {i}"))

