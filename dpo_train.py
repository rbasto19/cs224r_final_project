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

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)


def collate_fn(mix_batch):
    batch_win, batch_lose, protein_batch = list(zip(*mix_batch))
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

    for btc_idx in range(len(batch_win)):
        input_win_len = len(batch_win[btc_idx])
        input_lose_len = len(batch_lose[btc_idx])
        input_win_ids.append(batch_win[btc_idx])
        input_lose_ids.append(batch_lose[btc_idx])
        input_win_ids[btc_idx].extend([tokenizer.pad_token_id] * (max_win_input_len - input_win_len))
        input_lose_ids[btc_idx].extend([tokenizer.pad_token_id] * (max_lose_input_len - input_lose_len))

        # padding protein
        protein_ids[btc_idx, :len(protein_batch[btc_idx]), :] = protein_batch[btc_idx]

    return torch.tensor(input_win_ids, dtype=torch.long), torch.tensor(input_lose_ids, dtype=torch.long), torch.tensor(protein_ids, dtype=torch.float32)


def data_loader(args, train_data_win, train_data_lose, matrix_protein, tokenizer, shuffle):
    data_list = []
    for ind, (data_i_win, data_i_lose) in tqdm(enumerate(zip(train_data_win, train_data_lose))):
        # data_i = data_i.replace('GEO', '')
        data_i_win = '<|beginoftext|> <|mask:0|> <|mask:0|> ' + data_i_win + ' <|endofmask|>'
        mol_ = [tokenizer.encode(data_i_win, truncation=False, max_length=200, return_special_tokens_mask=True,
                                 add_special_tokens=False)]
        data_i_lose = '<|beginoftext|> <|mask:0|> <|mask:0|> ' + data_i_lose + ' <|endofmask|>'
        mol_.append(tokenizer.encode(data_i_lose, truncation=False, max_length=200, return_special_tokens_mask=True,
                                 add_special_tokens=False))

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
    beta=0.1
):
    out = beta * (log_probs_win - log_probs_lose - (log_probs_win_ref - log_probs_lose_ref))
    return torch.mean(-1 * torch.log(torch.sigmoid(out)))

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
    # input_ids.extend(tokenizer.encode(begin_text + response_text + ' <|endofmask|>', add_special_tokens=False))
    # input_length = len(input_ids)
    begin_ids = []
    begin_ids.extend(tokenizer.encode(begin_text, add_special_tokens=False))
    begin_length = len(begin_ids)
    


    # input_tensor = torch.zeros(len(protein_batch), input_length).long()
    # input_tensor[:] = torch.tensor(input_ids)
    # Seq_list = []
    # log_probs = torch.zeros(batch_size).to(device)
    # finished = torch.zeros(batch_size, 1).byte().to(device)

    # protein_batch = torch.tensor(np.array(single_protein), dtype=torch.float32)
    # protein_batch = protein_batch.to(device)
    # protein_batch = protein_batch.repeat(batch_size, 1, 1)

    # inputs = input_tensor.to(device)

    outputs = model(input_ids, protein_batch)
    logits = outputs.logits[:, :-1, :] # only focus on next tokens within the sentence, excluding new "generated" one
    labels = input_ids[:, 1:] # indices of next tokens
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(logits, 2, labels.unsqueeze(-1)).squeeze(-1) # collect log_probs of correct next tokens
    
    log_probs = torch.sum(log_probs[:, begin_length-1:], dim=-1) # get log_probs only of generated text

    return log_probs


def train(dataloader, ref_model, train_model, args):
    # rank, world_size = setup(rank, world_size)
    # torch.cuda.set_device(rank)

    #tokenizer = ExpressionBertTokenizer('../pocket_generate/data/torsion_version/torsion_voc_pocket.csv')
    tokenizer = ExpressionBertTokenizer('/home/ubuntu/cs224r_project/data_2/torsion_version/torsion_voc_pocket.csv')

    # Prior.to(rank)
    # Agent.to(rank)

    # We dont need gradients with respect to reference model
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.RMSprop(train_model.parameters(), lr=1e-6)  # default lr=1e-4

    # protein_emb = next(Path(protein_dir).glob('*.pkl'))
    # single_protein = read_data(protein_emb)
    
    # sync batch normalization
    # Agent = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Agent)
    # run model on the rank pid
    # Agent = DDP(Agent, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    print("Model initialized, starting training...")
    train_model.train()
    ref_model.eval()
    for epoch in range(args.num_epochs):
        for mix_batch in dataloader:
            optimizer.zero_grad()
            # get ids of tokens of ligand, and protein matrix
            lig_batch_win, lig_batch_lose , protein_batch = mix_batch
            # protein_batch = torch.tensor(protein_batch)
            # lig_batch_win = torch.tensor(lig_batch_win)
            # lig_batch_lose = torch.tensor(lig_batch_lose)
            log_probs_win = get_log_probs(train_model, tokenizer, protein_batch, lig_batch_win)
            log_probs_lose = get_log_probs(train_model, tokenizer, protein_batch, lig_batch_lose)
            # print(log_probs_win.shape)
            # print(log_probs_lose.shape)
            with torch.no_grad():
                log_probs_win_ref = get_log_probs(ref_model, tokenizer, protein_batch, lig_batch_win)
                log_probs_lose_ref = get_log_probs(ref_model, tokenizer, protein_batch, lig_batch_lose)
            loss = dpo_loss(log_probs_win, log_probs_lose, log_probs_win_ref, log_probs_lose_ref)
            # print(loss.shape)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
            optimizer.step()
            print(loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--world-size', action='store', dest='world_size', type=int,
                        default=4)
    parser.add_argument('--num-epochs', action='store', dest='num_epochs', type=int,
                        default=1000)
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                        default=8,
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

    # contains list of protein matrices
    with open('protein_data.pkl', 'rb') as f:
        protein_matrix = pickle.load(f)
    # contains list of corresponding ligands in smiles + torsion format
    with open('win_data.pkl', 'rb') as f:
        mol_data_win = pickle.load(f)[:len(protein_matrix)]
    with open('lose_data.pkl', 'rb') as f:
        mol_data_lose = pickle.load(f)[:len(protein_matrix)]

    tokenizer = ExpressionBertTokenizer('/home/ubuntu/cs224r_project/data_2/torsion_version/torsion_voc_pocket.csv')
    train_dataloader = data_loader(args, mol_data_win, mol_data_lose, protein_matrix, tokenizer=tokenizer, shuffle=True)
    train(train_dataloader, ref_model, train_model, args)

    # args_list.append(Prior)
    # args_list.append(Agent)

    # mp.spawn(train_agent,
    #          args=args_list,
    #          nprocs=args.world_size,
    #          join=True)


