import torch
import pickle
from threading import Thread
import logging
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
    util
)

from vllm import LLM, SamplingParams
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)
from flask import Flask, request
from flask_ipfilter import IPFilter, Whitelist

from pathlib import Path

# logging setting
log_filename = 'web_api'
logger = logging.getLogger(log_filename)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
fileHandler = logging.FileHandler(
    f'{log_filename}.log',
    mode='w',
    encoding='utf-8'
)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

# Display logging content simultaneously in terminal.
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)


'''
Flask Web API
'''
# Create Flask object
app = Flask(__name__)

# Set whitelist
ip_filter = IPFilter(app, ruleset=Whitelist())
ip_filter.ruleset.permit("127.0.0.1")

'''
Initialize variables
'''
# Use dictionary to save the conversation history
sessions_history = {} 

# Set CPU or GPU (cuda:5 or mps)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:5'
elif torch.backends.mps.is_available():
    device = 'mps'
# model.to(device)

# Llama
model_id = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
tokenizer = AutoTokenizer.from_pretrained(model_id)

llm = LLM(
    model=model_id,
    tensor_parallel_size=1,
    max_model_len=8192,
)

sampling_params = SamplingParams(
    temperature=0.2,
    top_p=0.9,
    max_tokens=512,
)

# Acquire the tokenizer of the model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create TextIteratorStreamer object
streamer = TextIteratorStreamer(tokenizer=tokenizer)

# Load embedding models
bi_encoder = SentenceTransformer(
    'BAAI/bge-m3',
    device=device,
)
cross_encoder = CrossEncoder(
    'BAAI/bge-reranker-v2-m3',
    device=device
)

# Load embeddings and passages
emb_file_path = "emb.pkl"
with open(emb_file_path, "rb") as fIn:
    stored_data = pickle.load(fIn)
    passage_embeddings = stored_data['passage_embeddings']
    passages = stored_data['passages']
    del stored_data

# Receive the result after "semantic search + re-ranking"
def get_results(query, search_size, top_k, add_gt, gt):

    # Receive the embedding of the query
    question_embedding = bi_encoder.encode(
        query,
        batch_size=1,
        device=device
    )

    # Semantic search
    hits = util.semantic_search(
        question_embedding,
        passage_embeddings,
        top_k=search_size
    )
    hits = hits[0]

    # Use cross_encoder to rate all the retrieved documents
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Use Cross-Encoder (Re-ranker) to re-rank the retrieved results
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Store the retrieved results after re-ranking.
    results = []
    scores = []
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    for hit in hits[:top_k]:
        results.append(passages[hit['corpus_id']])
        scores.append(hit['cross-score'])

    # Record the retrieved ids
    ids = [hit['corpus_id'] for hit in hits[:top_k]]

    # Test effectness of citation: Add ground truth to the retrieved list
    if add_gt:

        # Find missing ground truth ids and add missing ids to ids
        missing = [x for x in gt if x not in ids]
        ids.extend(missing)

        # Add corresponding content and scores
        for id in missing:
            results.append(passages[id])
            scores.append(0.0) # placeholder


    return results, scores, ids



# Search for specific words
@app.route("/generate", methods=["POST"])
def generate():
    # Receive JSON data from the front end.
    data = request.json

    # Receive session id and user message
    session_id = data.get("session_id")
    message = data.get("message")
    multi_turn_conversation = data.get("multi_turn_conversation")
    rag = data.get("rag")
    top_k = data.get("top_k")

    # Switch of citation function
    citation = data.get("citation")

    # test the effectness of citation
    gt = data.get("ground_truth")
    add_gt = data.get("add_gt")

    retrieved_ids = []

    # If rag = True, start retrieving (semantic search + re-ranking)
    if rag:
        # Set retrieval parameters.
        search_size = 100

        # Recerive retrieval results
        results, scores, retrieved_ids = get_results(message, search_size, top_k, add_gt, gt)


        # Turn results to text
        knowledge = ''
        i = 0
        for index, context in enumerate(results):
            knowledge += f"{index+1}) [id: {retrieved_ids[i]}] {context}\n"
            i += 1

        def load_prompt(file_path: str, knowledge: str, message: str) -> str:
            template = Path(file_path).read_text(encoding="utf-8")
            return template.format(knowledge=knowledge, message=message)

        # Create user prompt        
        prompt_file = "./prompts/with_citation.txt" if citation else "./prompts/without_citation.txt"
        message = load_prompt(prompt_file, knowledge, message)
        

        # See user prompt
        logger.info(message)
    
    # If session id not in conversation history, create a new conversation record for this session_id
    # To forget previous conversations, also create a new session record.
    if session_id not in sessions_history or multi_turn_conversation == False:

        # for MusiQue
        sessions_history[session_id] = [
            {"role": "system", "content": "AI chatbot with capability of answering multi-hop question."},
        ]

    # Put the user's message in the conversation record
    sessions_history[session_id].append({"role": "user", "content": message})

    # Store retrieved_ids in Flask's context
    request.environ['retrieved_ids'] = retrieved_ids


    # Generate reponses
    def generate_responses():

        # Convert multi-turn conversations in the record from dict to text to help the model generate a response
        tokenized_chat = tokenizer.apply_chat_template(
            conversation=sessions_history[session_id],
            add_generation_prompt=True,
            tokenize=False
        )

        # First yield retrieved IDs
        yield f"[[RETRIEVED_IDS]] {retrieved_ids}\n"

        # Generate response using vLLM
        outputs = llm.generate(prompts=[tokenized_chat], sampling_params=sampling_params)

        generated_text = outputs[0].outputs[0].text

        # Stream output token-by-token
        for token in generated_text:
            yield token

        logger.info(generated_text)
        logger.info('\n\n')

        # Save model reply to session
        sessions_history[session_id].append({"role": "assistant", "content": generated_text})
    
    return app.response_class(generate_responses(), mimetype='text/plain')


# Main
if __name__ == '__main__':
    app.debug = False
    app.json.ensure_ascii = False
    app.run(
        host='127.0.0.1', # 0.0.0.0
        port=5004
    )

