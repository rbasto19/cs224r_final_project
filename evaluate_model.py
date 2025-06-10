import argparse
import warnings
import numpy as np
import pandas as pd
import time
import os
import glob
from pathlib import Path
from tqdm import tqdm
import pickle
import torch
import torch.nn.functional as F
# import torch.distributed as dist
# import torch.multiprocessing as mp
from ada_model import Token3D
from torch.nn.parallel import DistributedDataParallel as DDP
from bert_tokenizer import ExpressionBertTokenizer
from transformers import GPT2Config
# from smi_torsion_2_molobj import construct_molobj
# from reward_score import scoring
# from MCMG_utils.data_structs import Experience
# from utils import Variable, unique, read_data, decode
from torch.utils.data import Dataset, DataLoader
import wandb
from utils.utils import cal_loss_and_accuracy, gce_loss_and_accuracy
from gen import predict, decode
from reinforce.reward_score import scoring, my_eval
from reinforce.smi_torsion_2_molobj import construct_molobj
from rdkit.Chem import AllChem, DataStructs
from datetime import datetime
from copy import deepcopy
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem.FilterCatalog import *
from dpo_train import MyDataset, data_loader, obey_lipinski, collate_fn, get_diversity
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# If TensorBoard is available
# from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda") if torch.cuda.is_available() else None
if not device:
    raise TimeoutError('No GPU detected.')

Ada_config = GPT2Config(
    architectures=["GPT2LMHeadModel"],
    model_type="GPT2LMHeadModel",
    vocab_size=836,
    n_positions=380,
    n_ctx=380,  # max length
    n_embd=768,
    n_layer=12,
    n_head=8,

    task_specific_params={
        "text-generation": {
            "do_sample": True,
            "max_length": 380
        }
    }
)

def evaluate(model, tokenizer, dataloader, args, step, protein_dirs, protein_encodings, timestamp):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = GPT2LMHeadModel.from_pretrained(args.save_model_path)
    # model.load_state_dict(torch.load('final_model_early_stop.pt'))

    # model.to(device)
    model.eval()
    loss_list = []
    vina_list = []
    qed_list = []
    sa_list = []
    lipinski_score = []
    valid = 0
    total = 0
    batch_steps = 0
    # early_stopping = EarlyStopping(patience=20, verbose=False)
    print("Eval...")
    with torch.no_grad():
        for mix_batch in dataloader:
            # NOTE: interesting, since this is not the sampled, but rather kinda
            #  like loss from just doing next token on validation set? Doesn't 
            # seem right. Better idea is to report the binding affinity of generated molecule.
            # so I first have to generate, then score. So idea is to create file
            # with list of protein dirs, generate candidate ligand, then score the
            # ligand (we don't use the "best ligand" here since we want to measure
            # the affinity of generated molecules )
            batch, _, _, _, protein_batch = mix_batch
            batch_steps += 1
            batch = batch.to(device)
            protein_batch = protein_batch.to(device)
            outputs = model(batch, protein_batch)
            
            loss, _ = gce_loss_and_accuracy(outputs, batch.to(device), device)
            loss_list.append(float(loss))
            # acc_list.append(float(acc))

    # now compute reward in generated ligand

    # generate
    gen_ligands = []
    list_ligands = []
    ligands_for_val = []
    print("Predicting...")
    for protein in tqdm(protein_encodings):
        predictions = predict(model, tokenizer, 5, protein)
        decoded_preds = []
        for prediction in predictions:
            decoded_preds.append(decode(prediction))
        # print(decoded_preds)
        gen_ligands.append(decoded_preds)
        ligands_for_val.append(decoded_preds[:1])
        list_ligands += decoded_preds
    
    print("Computing diversity...")
    div_list = []
    for ligands in gen_ligands:
        div_list.append(get_diversity(ligands))
    diversity = np.mean(div_list)

    print("Computing Lipinski...")
    all_mols = construct_molobj(list_ligands)
    for mol in all_mols:
        if mol is not None:
            lipinski_score.append(obey_lipinski(mol))
    
    print("Scoring...")
    for protein_dir, smiles_torsions in tqdm(zip(protein_dirs, ligands_for_val), total=len(protein_dirs)):
        mols = construct_molobj(smiles_torsions)
        protein_file = protein_dir.split('/')[-1]
        ligand_file = protein_dir.split('_pocket10.pdb')[0] + '.sdf'
        scores = my_eval(mols, '/'.join(protein_dir.split('/')[:-1]), protein_file, ligand_file)
        if scores['vina'] != None:
            vina_list.extend(scores['vina'])
            qed_list.extend(scores['qed'])
            sa_list.extend(scores['sa'])
        valid += scores['valid_count']
        total += scores['total_count']
    with open(f"metrics/{timestamp}_{step}.pkl", 'wb') as f:
        pickle.dump({
            "run_name": wandb.run.name,
            "beta": args.beta,
            'step': step,
            "val_loss": np.mean(loss_list),
            "examples_seen": step * args.batch_size,
            "valid_fraction": valid / total,
            "diversity": diversity,
            "affinity": vina_list,
            "qed": qed_list,
            "sa": sa_list,
            "lipinski": lipinski_score
        }, f)
    # print("val_loss: {},".format(np.mean(loss_list)))
    # print("avg_score: {},".format(np.mean(score_list)))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--world-size', action='store', dest='world_size', type=int,
                        default=4)
    parser.add_argument('--num-epochs', action='store', dest='num_epochs', type=int,
                        default=1000)
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                        default=16,
                        help='Batch size in a single device. Remember that total batch size = batch-size * world-size')
    parser.add_argument('--max-length', action='store', dest='max_length', type=int,
                        default=50)
    parser.add_argument('--sigma', action='store', dest='sigma', type=int,
                        default=60)
    parser.add_argument('--experience-replay', type=int, default=0)
    parser.add_argument('--restore-from', default='/home/ubuntu/cs224r_project/token_mol/Token-Mol/Trained_model/pocket_generation.pt',
                         help='Path for loading the model.')
    parser.add_argument('--protein-dir', action='store', dest='protein_dir',
                        default='./usecase_protein_embedding/CDK4',
                        help='Path where store protein target informations.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--lr', default=1e-6)
    parser.add_argument('--regularize', default=1, type=int)
    parser.add_argument('--reg_type', default="alidiff")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # parser.add_argument('--save-file-path', action='store', dest='save_dir',
    # help='Path where results and model are saved. Default is data/results/run_<datetime>.')

    args = parser.parse_args()
    args_list = list(vars(args).values())

    model = Token3D(pretrain_path='/home/ubuntu/cs224r_project/token_mol/Token-Mol/Pretrained_model', config=Ada_config)
    restore_from = args.restore_from
    
    ref_dict = {key.replace("module.", ""): value for key, value in
                        torch.load(restore_from, map_location='cuda').items()}
    train_dict = {key.replace("module.", ""): value for key, value in
                        torch.load(restore_from, map_location='cuda').items()}
    model.load_state_dict(ref_dict)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model.to(device)
    # contains list of protein matrices
    with open('protein_data_folder/protein_data.pkl', 'rb') as f:
        protein_matrix = pickle.load(f)
    # contains list of corresponding ligands in smiles + torsion format
    with open('win_data.pkl', 'rb') as f:
        mol_data_win = pickle.load(f)
    with open('lose_data.pkl', 'rb') as f:
        mol_data_lose = pickle.load(f)
    with open('protein_filename.pkl', 'rb') as f:
        protein_filenames = pickle.load(f)
    protein_dirs = []
    for protein in protein_filenames:
        protein_dirs.append('protein_data_folder/'+protein)
    
    with open('dpo_affinities.pkl', 'rb') as f:
        dpo_affinities = pickle.load(f)

    affinities_win = []
    affinities_lose = []
    for affinities in dpo_affinities:
        affinities_win.append(min(affinities))
        affinities_lose.append(max(affinities))
    
    val_size = 10
    wandb.init(
        project='dpo_train_224r',
        config={
            'batch_size': args.batch_size,
            'beta': args.beta,
            'lr': args.lr,
            "val_size": val_size,
            "train_size": len(protein_dirs),
            "regularize": args.regularize,
            "reg_type": args.reg_type
        }
    )
    tokenizer = ExpressionBertTokenizer('/home/ubuntu/cs224r_project/data_2/torsion_version/torsion_voc_pocket.csv')
    train_dataloader = data_loader(args, mol_data_win[val_size:], mol_data_lose[val_size:], affinities_win[val_size:], affinities_lose[val_size:], protein_matrix[val_size:], tokenizer=tokenizer, shuffle=True)
    val_dataloader = data_loader(args, mol_data_win[:val_size], mol_data_lose[:val_size], affinities_win[:val_size], affinities_lose[:val_size], protein_matrix[:val_size], tokenizer=tokenizer, shuffle=True)
    # train(train_dataloader, val_dataloader, ref_model, train_model, args, protein_dirs[:val_size], protein_matrix[:val_size])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
