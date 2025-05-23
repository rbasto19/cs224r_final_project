import subprocess
import pickle
import shlex
from tqdm import tqdm
import sys
sys.path.append('/home/ubuntu/cs224r_project/utils/')
sys.path.append('/home/ubuntu/cs224r_project/AliDiff/datasets')
import datasets
from rdkit import Chem
from standardization import get_torsion_angles, GetDihedral

def sdf_to_smiles_torsion(sdf_file, smiles):
    supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)
    molecule = None
    for mol in supplier:
        if mol is None:
            raise Exception()
            continue
        if mol.GetNumConformers() == 0:
            raise Exception()
            print("No conformers found in this molecule.")
            continue
        conf = mol.GetConformer()
        molecule = mol
        break
    torsion_list = []
    for torsion in get_torsion_angles(molecule):
        torsion_list.append(f"{GetDihedral(conf, torsion):.2f}")
    smiles_torsion = ' '.join(smiles)
    smiles_torsion += ' GEO '
    smiles_torsion += ' '.join(torsion_list)
    return smiles_torsion

with open('/home/ubuntu/cs224r_project/AliDiff/dpo_train_set.pkl', 'rb') as f:
    dpo_train_set = pickle.load(f)

data_path = '/home/ubuntu/cs224r_project/AliDiff/data/crossdocked_v1.3_rmsd1.0_pocket10'
win_data = []
lose_data = []

for i, pl_pair in tqdm(enumerate(dpo_train_set), total=len(dpo_train_set)):
    win_lig_filename = pl_pair.ligand_filename
    win_lig_smiles = pl_pair.ligand_smiles
    win_data.append(sdf_to_smiles_torsion(win_lig_filename, win_lig_smiles))
    lose_lig_filename = pl_pair.ligand_filename2
    lose_lig_smiles = pl_pair.ligand_smiles2
    lose_data.append(sdf_to_smiles_torsion(lose_lig_filename, lose_lig_smiles))

with open('win_data.pkl', 'wb') as f:
    pickle.dump(win_data, f) 
with open('lose_data.pkl', 'wb') as f:
    pickle.dump(lose_data, f) 


