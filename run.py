import requests as req
import pandas as pd
import json as js
import os
import csv
import math
import re
import time

headers = {
    "Content-Type": "application/json"
}

sess_id = 0
queries = []
correct_ids = []
df = None

# Extract content of data file
def extract_content(df):
    global queries, correct_ids
    queries, correct_ids = [], []
    for idx, row in df.iterrows():
        queries.append(row["Query"])
        correct_ids.append(list(map(int, row["Retrieved_id"].split(","))))

# Talk with LLM
def chat(message, num_retrieved, cite, gt, add_gt, session_id="sess_5566"):
    payload = {
        "session_id": session_id,
        "message": message,
        "multi_turn_conversation": False,
        "rag": True,
        "top_k": num_retrieved,
        "citation": cite,
        "ground_truth": gt,
        "add_gt":  add_gt,
    }

    retrieved_ids = None
    generated_text = ""

    # Extract id list and generated text
    with req.post(url='http://127.0.0.1:5004/generate', stream=False, headers=headers, json=payload) as res:
        full_text = res.text
        
        if "[[RETRIEVED_IDS]]" in full_text:
            parts = full_text.split("[[RETRIEVED_IDS]]")
            lines = parts[1].split("\n", 1)  # split into ID line and the rest

            id_line = lines[0].strip()
            try:
                retrieved_ids = eval(id_line)
            except Exception as e:
                print(f"[Error parsing retrieved IDs: {e}]")
                retrieved_ids = []

            generated_text = lines[1] if len(lines) > 1 else ""
        else:
            # If no retrieved ids tag, treat entire text as generated_text
            generated_text = full_text

    return retrieved_ids, generated_text

# Ask all the questions and save results
def ask(iter, num_retrieved=10, cite=True, add_gt=False):
    global sess_id, df
    all_results = [[] for _ in range(iter)]

    for idx, query in enumerate(queries):
        print(f"\n--- Processing query {idx+1}/{len(df)} ---")

        reasoning = ""
        retrieved_ids = [[] for _ in range(iter)]
        t = 0
        for i in range(iter):
            question = query + reasoning
            sess_id += idx

            # Get the response from rag and record the required time
            t0 = time.time()
            retrieved_ids[i], response = chat(question, num_retrieved, cite, correct_ids[idx], add_gt, f"sess_{sess_id}")
            t1 = time.time()
            t += t1 - t0

            # Extract citation ids from the response and reorder the list
            retrieved_ids[i] = extract_reorder_ids(response, retrieved_ids[i])

            reasoning = f" {response.split('The answer is:')[0] if 'The answer is:' in response else response}"
            
            result = {"Initial query": query}
            if i > 0:
                result["Expanded"] = question
            
            result["retrieved_ids"] = retrieved_ids[:i+1]
            result["response"] = response
            result["time"] = t
            all_results[i].append(result)

    for i, results in enumerate(all_results):
        with open(f"results_iter{i+1}.json", "w") as f:
            js.dump(results, f, indent=2)

    return all_results

# Extract the citation ids and move the ids that are found in the text to the front in the order they appear in old list
def extract_reorder_ids(text, old_list):
    # Extract all IDs from the text
    id_groups = re.findall(r'id:\s*([\d\s,]+)', text)
    mentioned_ids = set()

    for group in id_groups:
        for id_str in group.split(','):
            id_str = id_str.strip()
            if id_str.isdigit():
                mentioned_ids.add(int(id_str))

    # Reorder old_list: first the ones mentioned, then the rest
    front = [x for x in old_list if x in mentioned_ids]
    back = [x for x in old_list if x not in mentioned_ids]
    return front + back

# Combine the retrieved ids lists and calculate NDCG
def combine_ndcg(retrieved_ids, correct_ids, k=None):
    doc_scores = {}
    query_expanded_pos = {}

    # Build position map for query-expanded list (last iteration)
    if len(retrieved_ids) > 1:
        for pos, doc_id in enumerate(retrieved_ids[-1]):
            query_expanded_pos[doc_id] = pos

    # Calculate the score of every retrieved document
    for i, ids in enumerate(retrieved_ids):
        for rank, id in enumerate(ids):
            rank_discount = 1 / math.log2(rank + 2)
            if id not in doc_scores:
                doc_scores[id] = 0
            doc_scores[id] += rank_discount  # Add rank weight per appearance

    # Rerank all docs based on score (descending)
    sorted_docs = sorted(
        doc_scores.items(),
        key=lambda x: (
            -x[1],
            query_expanded_pos.get(x[0], float('inf'))
        )
    )
    reranked = [doc for doc, _ in sorted_docs]

    # top-k truncation (optional)
    if k:
        reranked = reranked[:k]

    # Compute DCG
    dcg = 0.0
    for i, id in enumerate(reranked):
        if id in correct_ids:
            dcg += 1 / math.log2(i + 2)

    # Compute ideal DCG
    idcg = sum(1 / math.log2(i + 2) for i in range(len(correct_ids)))

    # normalize
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return reranked, ndcg


# Evaluate rag performance and save result
def evaluate(all_results, question_set, filename="metrics.csv"):
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['iteration', 'avg_ndcg', 'avg_ndcg_filtered', 'avg_recall', 'filtered_size',
                             'avg_combined_list_length', 'question_set', 'time'])
        
        num_questions = len(all_results[0])
        num_iters = len(all_results)

        # Track per-iteration ndcg/recall/ndcg_filtered
        iter_ndcgs = [[] for _ in all_results]
        iter_recalls = [[] for _ in all_results]
        iter_ndcgs_filtered = [[] for _ in all_results]
        iter_times = [[] for _ in all_results]
        iter_combined_len = [[] for _ in all_results]

        for q in range(num_questions):

            # Metrics for current quesiton only
            recalls_for_q = []
            ndcgs_for_q = []

            for iter in range(num_iters):
                result = all_results[iter][q]
                
                unique_ids, ndcg = combine_ndcg(result["retrieved_ids"], correct_ids[q])

                pred_ids = set(unique_ids)
                gt_ids = set(correct_ids[q])

                tp = len(pred_ids.intersection(gt_ids))
                recall = tp / len(gt_ids) if gt_ids else 0.0

                # Append to global metrics
                iter_ndcgs[iter].append(ndcg)
                iter_recalls[iter].append(recall)
                iter_times[iter].append(result["time"])
                iter_combined_len[iter].append(len(unique_ids))

                # Keep per-question copy for later filtered NDCG
                recalls_for_q.append(recall)
                ndcgs_for_q.append(ndcg)

            # If any iteration has recall == 1, include its NDCG in filtered list
            if any(r == 1 for r in recalls_for_q):
                for iter_idx in range(num_iters):
                    iter_ndcgs_filtered[iter_idx].append(ndcgs_for_q[iter_idx])
        
        #  Report and write results per iteration
        for iter_idx in range(num_iters):
            avg_ndcg = sum(iter_ndcgs[iter_idx]) / len(iter_ndcgs[iter_idx])
            avg_recall = sum(iter_recalls[iter_idx]) / len(iter_recalls[iter_idx])

            filtered_size = len(iter_ndcgs_filtered[iter_idx])
            avg_ndcg_filtered = (
                sum(iter_ndcgs_filtered[iter_idx]) / filtered_size
                if iter_ndcgs_filtered[iter_idx]
                else 0
            )

            avg_time = sum(iter_times[iter_idx]) / len(iter_times[iter_idx])
            avg_combined_len = sum(iter_combined_len[iter_idx]) / len(iter_combined_len[iter_idx])

            print(f"[Iter {iter_idx + 1}] Avg NDCG: {avg_ndcg:.4f}")
            print(f"[Iter {iter_idx + 1}] Avg NDCG Filtered: {avg_ndcg_filtered:.4f}")
            print(f"[Iter {iter_idx + 1}] Size after filtering: {filtered_size}")
            print(f"[Iter {iter_idx + 1}] Avg Recall: {avg_recall:.4f}")
            print(f"[Iter {iter_idx + 1}] Avg combined list length: {avg_combined_len:.4f}")
            print(f"[Iter {iter_idx + 1}] Avg Time: {avg_time:.4f}")
            print()

            writer.writerow([iter_idx + 1, round(avg_ndcg, 4), round(avg_ndcg_filtered, 4), round(avg_recall, 4)
                             , filtered_size, round(avg_combined_len, 1), question_set, round(avg_time, 4)])

def run(iter):
    global df
    for i in range(2, 5):
        print(f"\n{i}hop question set:")

        # Quesiton file
        df = pd.read_csv(f'./MusiQue_cleaned/{i}hop_ans_dev.csv') 
        extract_content(df)

        results = ask(iter, add_gt=True)

        evaluate(results, question_set=f'{i}hop', filename="metrics_add_gt.csv")

def test(iter):
    global df
    df = pd.read_csv(f'test.csv')
    extract_content(df)

    results = ask(iter, add_gt=True)

    evaluate(results, question_set=f'test', filename="metrics_add_gt.csv")

# main
if __name__ == '__main__':
    
    # run
    run(iter=2)

    # test
    # test(iter=2)