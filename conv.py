import sqlite3
import pickle
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

# Set CPU or GPU (cuda:0 or mps)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'
elif torch.backends.mps.is_available():
    device = 'mps'

# Load the BGE-M3 model
bi_encoder = SentenceTransformer('BAAI/bge-m3', device=device)

# REceive all the text data
passages = []

# For MusiQue database (DB.csv)
df = pd.read_csv('./MusiQue_cleaned/DB.csv')

# Count the number of tokens for each row
for index, row in df.iterrows():
    info = f"(Title: {row['title']}): {row['content']}"
    passages.append(info)

# Create passage embedding
passage_embeddings = bi_encoder.encode(
    passages,
    batch_size=2,
    device=device,
    show_progress_bar=True
)

# Svae the embeddings and passages fromt the documents
emb_file_path = "emb.pkl"
with open(emb_file_path, "wb") as fOut:
    pickle.dump({
        'passage_embeddings': passage_embeddings,
        'passages': passages
    }, fOut, protocol=pickle.HIGHEST_PROTOCOL)