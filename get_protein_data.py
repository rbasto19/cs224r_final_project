import os
import pickle

protein_data = []
for file in os.listdir('/home/ubuntu/cs224r_project/encoder/embeddings'):
    with open(f'/home/ubuntu/cs224r_project/encoder/embeddings/{file}', 'rb') as f:
        protein_data.append(pickle.load(f)[0])

with open('protein_data.pkl', 'wb') as f:
    pickle.dump(protein_data, f)