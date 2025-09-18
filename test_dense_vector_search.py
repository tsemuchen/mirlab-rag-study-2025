'''
# Download sentence Transformers
$ pip install sentence-transformers
'''
import pickle
import torch
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
    util
)

'''
Set CPU or GPU (cuda:0 or mps)
'''
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'
elif torch.backends.mps.is_available():
    device = 'mps'

'''
Query setting
'''

# query for MusiQue
query = '''
What was the person who provided evidence to suggest the existence of the neutron a participant of?
A) The Human Genome Project
B) The Apollo Program
C) The Manhattan Project
D) The Bell Labs Transistor Team'''

print("Query:", query)



'''
Read the passages and the corresponding embeddings
'''
# Read embeddings and passages
emb_file_path = "emb.pkl"
with open(emb_file_path, "rb") as fIn:
    stored_data = pickle.load(fIn)
    passage_embeddings = stored_data['passage_embeddings']
    passages = stored_data['passages']
    del stored_data


print("=" * 80)


'''
Semantic Search phase
'''
##### Semantic Search #####
bi_encoder = SentenceTransformer('BAAI/bge-m3', device=device)

# Obtain query embedding
question_embedding = bi_encoder.encode(
    query,
    batch_size=1,
    device=device
)

# Number of documents retrieved through semantic search
SEARCH_SIZE = 100

# Acquire the top k documents
TOP_K = 5

# Semantic search
hits = util.semantic_search(
    question_embedding,
    passage_embeddings,
    top_k=SEARCH_SIZE
)
hits = hits[0]

print(f"Top-{TOP_K} Bi-Encoder Retrieval hits (Semantic Search)")
hits = sorted(hits, key=lambda x: x['score'], reverse=True)
for hit in hits[0:TOP_K]:
    print(f"\t{hit['score']:.3f}\t{passages[hit['corpus_id']]}")


print("=" * 80)


'''
Re-Ranking phase
'''
##### Re-Ranking #####
cross_encoder = CrossEncoder('BAAI/bge-reranker-large', device=device)

# Use cross_encoder to rate all the retrieved documents
cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
cross_scores = cross_encoder.predict(cross_inp)

# Use Cross-Encoder Re-ranker to re-rank the retrieved results
for idx in range(len(cross_scores)):
    hits[idx]['cross-score'] = cross_scores[idx]

print(f"Top-{TOP_K} Cross-Encoder Re-ranker hits (Relevance Score)")
hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
for hit in hits[0:TOP_K]:
    print(f"\t{hit['cross-score']:.3f}\t{passages[hit['corpus_id']]}")