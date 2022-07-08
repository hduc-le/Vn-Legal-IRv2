import os
import json
import pickle
import logging
from re import S
from unicodedata import name
import numpy as np
from tqdm import tqdm
from rank_bm25 import *
from utils import flatten2DList, load_json, load_parameter

import argparse

from vncorenlp import VnCoreNLP
from rank_bm25 import BM25Okapi

logging.basicConfig(
    format="%(asctime)s %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p %Z",
    level = logging.INFO
)

def output_scores(scores):

    for k in scores['accuracy@k']:
        logging.info("Accuracy@{}: {:.2f}%".format(k, scores['accuracy@k'][k]*100))

    for k in scores['precision@k']:
        logging.info("Precision@{}: {:.2f}%".format(k, scores['precision@k'][k]*100))

    for k in scores['recall@k']:
        logging.info("Recall@{}: {:.2f}%".format(k, scores['recall@k'][k]*100))
    
    for k in scores['f2-score@k']:
        logging.info("f2-score@{}: {:.2f}%".format(k, scores['f2-score@k'][k]*100))
    
    for k in scores['mrr@k']:
        logging.info("MRR@{}: {:.4f}".format(k, scores['mrr@k'][k]))
    
    for k in scores['ndcg@k']:
        logging.info("NDCG@{}: {:.4f}".format(k, scores['ndcg@k'][k]))
    
    for k in scores['map@k']:
        logging.info("MAP@{}: {:.4f}".format(k, scores['map@k'][k]))

def calculate_f2(precision, recall):        
    return (5 * precision * recall) / (4 * precision + recall + 1e-20)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stopword", default="manual", type=str)
    parser.add_argument("--model_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--raw_data", default="data", type=str, help="path to input data")
    parser.add_argument("--legal_path", default="generated_data/", type=str, help="path to save pair sentence directory")
    args = parser.parse_args()


    train_path = os.path.join(args.raw_data, "train_question_answer.json")
    training_data = load_json(train_path)

    training_items = training_data["items"]

    # with open(args.model_path, "rb") as bm_file:
    bm25 = load_parameter(args.model_path)

    doc_refers = load_parameter(os.path.join(args.legal_path, "doc_refers_saved.pkl"))

    doc_data = load_json(os.path.join("legal_dict.json"))
    corpus_ids = [id for id in doc_data]

    annotator = VnCoreNLP(
        args.word_segmenter, 
        annotators="wseg,pos,ner,parse", 
        max_heap_size="-Xmx2g"
    )

    save_pairs = []

    num_queries = len(training_items)
    
    accuracy_at_k = [1,3,5,10]
    precision_recall_at_k = [1,3,5,10]

    top_n = max(accuracy_at_k)

    num_hits_at_k = {k: 0 for k in accuracy_at_k}
    precisions_at_k = {k: [] for k in precision_recall_at_k}
    recall_at_k = {k: [] for k in precision_recall_at_k}
    f2_score_at_k = {k: [] for k in precision_recall_at_k}

    for idx, item in tqdm(enumerate(training_items)):
        if idx >= num_queries:
            continue

        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        tokenized_query = flatten2DList(annotator.tokenize(question))
        doc_scores = bm25.get_scores(tokenized_query)

        prediction_ids = np.argsort(doc_scores)[::-1][:top_n]
        pred_labels = [corpus_ids[id] for id in prediction_ids]

        query_relevant_docs = [article["law_id"] + "@" + article["article_id"] for article in relevant_articles]

        for k_val in accuracy_at_k:
            for pred_label in pred_labels[0:k_val]:
                if pred_label in query_relevant_docs:
                    num_hits_at_k[k_val] += 1
                    break
        
        # Precision and Recall@k
        for k_val in precision_recall_at_k:
            num_correct = 0
            for pred_label in pred_labels[0:k_val]:
                if pred_label in query_relevant_docs:
                    num_correct += 1

            precision = num_correct / k_val
            recall = num_correct / len(query_relevant_docs)
            f2_score = calculate_f2(precision, recall)

            precisions_at_k[k_val].append(precision)
            recall_at_k[k_val].append(recall)
            f2_score_at_k[k_val].append(f2_score)
    
    # Compute averages
    for k in num_hits_at_k:
        num_hits_at_k[k] /= len(num_queries)
    for k in precisions_at_k:
        precisions_at_k[k] = np.mean(precisions_at_k[k])
    for k in recall_at_k:
        recall_at_k[k] = np.mean(recall_at_k[k])
    for k in f2_score_at_k:
        f2_score_at_k[k] = np.mean(f2_score_at_k[k])

    scores = {'accuracy@k': num_hits_at_k, 
                'precision@k': precisions_at_k,
                'recall@k': recall_at_k, 
                'f2-score@k': f2_score_at_k}

    output_scores(scores)
    # close the sever
    annotator.close()