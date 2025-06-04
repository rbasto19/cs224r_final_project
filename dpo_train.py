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

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)

def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = Crippen.MolLogP(mol)
    rule_4 = (logp >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def collate_fn(mix_batch):
    batch_win, batch_lose, batch_aff_win, batch_aff_lose, protein_batch = list(zip(*mix_batch))
    input_win_ids = []
    input_lose_ids = []

    input_win_lens_list = [len(w) for w in batch_win]
    input_lose_lens_list = [len(w) for w in batch_lose]
    input_protein_len_list = [len(ww) for ww in protein_batch]

    max_win_input_len = max(input_win_lens_list)
    max_lose_input_len = max(input_lose_lens_list)
    max_protein_len = max(input_protein_len_list)

    # create a zero array for padding protein batch
    protein_ids = np.zeros((len(protein_batch), max_protein_len, len(protein_batch[0][0])),
                           dtype=protein_batch[0][0].dtype)

    aff_win_ids = []
    aff_lose_ids = []
    for btc_idx in range(len(batch_win)):
        input_win_len = len(batch_win[btc_idx])
        input_lose_len = len(batch_lose[btc_idx])
        input_win_ids.append(batch_win[btc_idx])
        input_lose_ids.append(batch_lose[btc_idx])
        input_win_ids[btc_idx].extend([tokenizer.pad_token_id] * (max_win_input_len - input_win_len))
        input_lose_ids[btc_idx].extend([tokenizer.pad_token_id] * (max_lose_input_len - input_lose_len))
        aff_win_ids.append(batch_aff_win[btc_idx])
        aff_lose_ids.append(batch_aff_lose[btc_idx])
        # padding protein
        protein_ids[btc_idx, :len(protein_batch[btc_idx]), :] = protein_batch[btc_idx]

    return torch.tensor(input_win_ids, dtype=torch.long), torch.tensor(input_lose_ids, dtype=torch.long), torch.tensor(aff_win_ids, dtype=torch.float32), torch.tensor(aff_lose_ids, dtype=torch.float32), torch.tensor(protein_ids, dtype=torch.float32)


def data_loader(args, train_data_win, train_data_lose, affinities_win, affinities_lose, matrix_protein, tokenizer, shuffle):
    data_list = []
    for ind, (data_i_win, data_i_lose, aff_win, aff_lose) in tqdm(enumerate(zip(train_data_win, train_data_lose, affinities_win, affinities_lose))):
        # data_i = data_i.replace('GEO', '')
        data_i_win = '<|beginoftext|> <|mask:0|> <|mask:0|> ' + data_i_win + ' <|endofmask|>'
        mol_ = [tokenizer.encode(data_i_win, truncation=False, max_length=200, return_special_tokens_mask=True,
                                 add_special_tokens=False)]
        data_i_lose = '<|beginoftext|> <|mask:0|> <|mask:0|> ' + data_i_lose + ' <|endofmask|>'
        mol_.append(tokenizer.encode(data_i_lose, truncation=False, max_length=200, return_special_tokens_mask=True,
                                 add_special_tokens=False))

        mol_.append(aff_win)
        mol_.append(aff_lose)
        mol_.append(matrix_protein[ind])
        
        data_list.append(mol_)

    dataset = MyDataset(data_list)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)

    return dataloader

def dpo_loss(
    log_probs_win, 
    log_probs_lose, 
    log_probs_win_ref, 
    log_probs_lose_ref,
    reward_win,
    reward_lose,
    response_len,
    beta=0.1,
    regularize=True,
):
    out = beta * (log_probs_win - log_probs_lose - (log_probs_win_ref - log_probs_lose_ref))
    out = torch.nn.functional.logsigmoid(out)
    # add regularization
    if regularize:
        # print("regularizing")
        # AliDiff regularization
        if response_len is None:
            reg = torch.sigmoid(-1 * (reward_win - reward_lose))
            out = reg * out + (1 - reg) * (1 - out)
        else: 
            # Llama-like regularization
            alpha = 0.2
            reg = alpha * log_probs_win / response_len
            out += reg
    return torch.mean(-1 * out)

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


def get_log_probs(model, tokenizer, protein_batch, input_ids):
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model.to(device)

    # input_ids = []
    begin_text = "<|beginoftext|> <|mask:0|> <|mask:0|> "
    begin_ids = []
    begin_ids.extend(tokenizer.encode(begin_text, add_special_tokens=False))
    begin_length = len(begin_ids)


    outputs = model(input_ids, protein_batch)
    logits = outputs.logits[:, :-1, :] # only focus on next tokens within the sentence, excluding new "generated" one
    labels = input_ids[:, 1:] # indices of next tokens
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(logits, 2, labels.unsqueeze(-1)).squeeze(-1) # collect log_probs of correct next tokens
    
    log_probs = torch.sum(log_probs[:, begin_length-1:], dim=-1) # get log_probs only of generated text

    return log_probs

def get_diversity(list_ligands):
    all_mols = construct_molobj(list_ligands)
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        for mol in all_mols if mol is not None]
    n = len(fps)
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sims.append(sim)
    avg_sim = np.mean(sims) if sims else 0
    return 1 - avg_sim

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
    # epoch_loss = np.mean(loss_list)
    # early_stopping(epoch_loss, model, args.early_stop_path)

    wandb.log({
        'step': step,
        "val_loss": np.mean(loss_list),
        "examples_seen": step * args.batch_size,
        "valid_fraction": valid / total,
        "diversity": diversity,
        "avg_affinity": np.mean(vina_list),
        "avg_qed": np.mean(qed_list),
        "avg_sa": np.mean(sa_list),
        "avg_lipinski": np.mean(lipinski_score),
    })
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

def train(dataloader, eval_loader, ref_model, train_model, args, protein_dirs, protein_encodings):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # rank, world_size = setup(rank, world_size)
    # torch.cuda.set_device(rank)

    #tokenizer = ExpressionBertTokenizer('../pocket_generate/data/torsion_version/torsion_voc_pocket.csv')
    tokenizer = ExpressionBertTokenizer('/home/ubuntu/cs224r_project/data_2/torsion_version/torsion_voc_pocket.csv')

    # Prior.to(rank)
    # Agent.to(rank)

    # We dont need gradients with respect to reference model
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.RMSprop(train_model.parameters(), lr=args.lr)  # default lr=1e-4

    # protein_emb = next(Path(protein_dir).glob('*.pkl'))
    # single_protein = read_data(protein_emb)
    
    # sync batch normalization
    # Agent = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Agent)
    # run model on the rank pid
    # Agent = DDP(Agent, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    print("Model initialized, starting training...")
    train_model.train()
    ref_model.eval()
    val_freq = len(dataloader)
    step = 0
    for epoch in range(args.num_epochs):
        evaluate(train_model, tokenizer, eval_loader, args, step, protein_dirs, protein_encodings, timestamp)
        for mix_batch in tqdm(dataloader, total=len(dataloader)):           
            optimizer.zero_grad()
            # get ids of tokens of ligand, and protein matrix
            lig_batch_win, lig_batch_lose, aff_win, aff_lose, protein_batch = mix_batch
            protein_batch = protein_batch.to(device)
            lig_batch_win = lig_batch_win.to(device)
            lig_batch_lose = lig_batch_lose.to(device)
            aff_win = aff_win.to(device)
            aff_lose = aff_lose.to(device)
            log_probs_win = get_log_probs(train_model, tokenizer, protein_batch, lig_batch_win)
            log_probs_lose = get_log_probs(train_model, tokenizer, protein_batch, lig_batch_lose)
            # print(log_probs_win.shape)
            # print(log_probs_lose.shape)
            with torch.no_grad():
                log_probs_win_ref = get_log_probs(ref_model, tokenizer, protein_batch, lig_batch_win)
                log_probs_lose_ref = get_log_probs(ref_model, tokenizer, protein_batch, lig_batch_lose)
            response_len = torch.tensor([len(ids) for ids in lig_batch_win]).to(device) if args.reg_type == 'llama' else None
            loss = dpo_loss(log_probs_win, log_probs_lose, log_probs_win_ref, log_probs_lose_ref, aff_win, aff_lose, response_len=response_len, beta=args.beta, regularize=args.regularize)
            # print(loss.shape)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
            optimizer.step()
            # print(loss)
            wandb.log({
                'step': step,
                "dpo_loss": loss.item(),
                "examples_seen": step * args.batch_size
            })
            step += 1
        # evaluate(train_model, tokenizer, eval_loader, args, step, protein_dirs, protein_encodings)


if __name__ == "__main__":
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
    parser.add_argument('--reg_type', default="llama")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # parser.add_argument('--save-file-path', action='store', dest='save_dir',
    # help='Path where results and model are saved. Default is data/results/run_<datetime>.')

    args = parser.parse_args()
    args_list = list(vars(args).values())

    ref_model = Token3D(pretrain_path='/home/ubuntu/cs224r_project/token_mol/Token-Mol/Pretrained_model', config=Ada_config)
    train_model = Token3D(pretrain_path='/home/ubuntu/cs224r_project/token_mol/Token-Mol/Pretrained_model', config=Ada_config)
    restore_from = args.restore_from
    
    ref_dict = {key.replace("module.", ""): value for key, value in
                        torch.load(restore_from, map_location='cuda').items()}
    train_dict = {key.replace("module.", ""): value for key, value in
                        torch.load(restore_from, map_location='cuda').items()}
    ref_model.load_state_dict(ref_dict)
    train_model.load_state_dict(train_dict)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    ref_model.to(device)
    train_model.to(device)
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
    train(train_dataloader, val_dataloader, ref_model, train_model, args, protein_dirs[:val_size], protein_matrix[:val_size])

    # args_list.append(Prior)
    # args_list.append(Agent)

    # mp.spawn(train_agent,
    #          args=args_list,
    #          nprocs=args.world_size,
    #          join=True)


