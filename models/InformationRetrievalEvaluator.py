import torch
import numpy as np
from torch import Tensor
from utils import *
from typing import List, Callable, Dict, Set
from tqdm.notebook import trange

class InformationRetrievalEvaluator:
    """
    This class evaluates an Information Retrieval (IR) setting.
    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
    """
    def __init__(self,
                 queries: Dict[str, str],  # qid => query
                 corpus: Dict[str, str],  # cid => doc
                 relevant_docs: Dict[str, Set[str]],  # qid => Set[cid]
                 corpus_chunk_size: int = 200,
                 mrr_at_k: List[int] = [1,3,5,10],
                 ndcg_at_k: List[int] = [1,3,5,10],
                 accuracy_at_k: List[int] = [1,3,5,10],
                 precision_recall_at_k: List[int] = [1,3,5,10],
                 map_at_k: List[int] = [1,3,5,10],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 score_functions: List[Callable[[Tensor, Tensor], Tensor]] = {
                        'cos_sim': cos_sim
                    },  # Score function, higher=more similar
                 main_score_function: str = None,
                 eval_mode: str = 'full_id' # full_id means the precision of recommend is law id and article id, law_id is otherwise
                 ):
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]
        
        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k
        self.eval_mode = eval_mode
        
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
    
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = main_score_function

    def compute_metrices(self, model, corpus_model = None, corpus_embeddings: Dict[str,Tensor] = None) -> Dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.accuracy_at_k), max(self.precision_recall_at_k), max(self.map_at_k))

        # Compute embedding for the queries
        model_query_embeddings = model.encode(self.queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)
    

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(model_query_embeddings))]

        #Iterate over chunks of the corpus
        for corpus_start_idx in trange(0, len(self.corpus), self.corpus_chunk_size, desc='Corpus Chunks', disable=not self.show_progress_bar):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            #Encode chunk of corpus
            if corpus_embeddings is None:
                sub_model_corpus_embeddings = corpus_model.encode(self.corpus[corpus_start_idx:corpus_end_idx], show_progress_bar=False, batch_size=self.batch_size, convert_to_tensor=True)
            else:
                sub_model_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]
    
    
            #Compute cosine similarites
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(model_query_embeddings, sub_model_corpus_embeddings)

                #Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False)
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(model_query_embeddings)):
                    for sub_corpus_id, score in zip(pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]):
                        full_id = self.corpus_ids[corpus_start_idx+sub_corpus_id]
                        law_id, article_id = full_id.split("@")

                        if law_id.endswith("nd-cp"):
                            law_id = law_id.replace("nd-cp", "nđ-cp")
                        if law_id.endswith("nđ-"):
                            law_id = law_id.replace("nđ-", "nđ-cp")
                        if law_id.endswith("nð-cp"):
                            law_id = law_id.replace("nð-cp", "nđ-cp")
                        if law_id == "09/2014/ttlt-btp-tandtc-vksndtc":
                            law_id = "09/2014/ttlt-btp-tandtc-vksndtc-btc"
                        
                        queries_result_list[name][query_itr].append({'law_id': law_id, 'article_id': article_id, 'score': score})

        print("Queries: {}".format(len(self.queries)))
        print("Corpus: {}".format(len(self.corpus)))

        #Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        #Output
        for name in self.score_function_names:
            print("Score-Function: {}".format(name))
            self.output_scores(scores[name])
        return scores, queries_result_list

    def compute_metrics(self, queries_result_list: List[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        f2_score_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}
        
        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            if self.eval_mode == 'full_id':
                # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
                for k_val in self.accuracy_at_k:
                    for hit in top_hits[0:k_val]:
                        full_id = hit['law_id']+'@'+hit['article_id']
                        if full_id in query_relevant_docs:
                            num_hits_at_k[k_val] += 1
                            break

                # Precision and Recall@k
                for k_val in self.precision_recall_at_k:
                    num_correct = 0
                    for hit in top_hits[0:k_val]:
                        full_id = hit['law_id']+'@'+hit['article_id']
                        if full_id in query_relevant_docs:
                            num_correct += 1

                    precision = num_correct / k_val
                    recall = num_correct / len(query_relevant_docs)
                    f2_score = self.calculate_f2(precision, recall)

                    precisions_at_k[k_val].append(precision)
                    recall_at_k[k_val].append(recall)
                    f2_score_at_k[k_val].append(f2_score)
                # MRR@k
                for k_val in self.mrr_at_k:
                    for rank, hit in enumerate(top_hits[0:k_val]):
                        full_id = hit['law_id']+'@'+hit['article_id']
                        if full_id in query_relevant_docs:
                            MRR[k_val] += 1.0 / (rank + 1)
                            break

                # NDCG@k
                for k_val in self.ndcg_at_k:
                    predicted_relevance = [1 if top_hit['law_id']+'@'+top_hit['article_id'] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]]
                    true_relevances = [1] * len(query_relevant_docs)

                    ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
                    ndcg[k_val].append(ndcg_value)

                # MAP@k
                for k_val in self.map_at_k:
                    num_correct = 0
                    sum_precisions = 0

                    for rank, hit in enumerate(top_hits[0:k_val]):
                        full_id = hit['law_id']+'@'+hit['article_id']
                        if full_id in query_relevant_docs:
                            num_correct += 1
                            sum_precisions += num_correct / (rank + 1)

                    avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                    AveP_at_k[k_val].append(avg_precision)

            else: # law_id

                # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
                for k_val in self.accuracy_at_k:
                    for hit in top_hits[0:k_val]:
                        checked_hit = False
                        for query_id in query_relevant_docs:
                            query_law_id = query_id.split("@")[0]
                            if hit['law_id'] == query_law_id:
                                num_hits_at_k[k_val] += 1
                                checked_hit = True
                                break
                        if checked_hit:
                            break
                
                # Precision and Recall@k
                for k_val in self.precision_recall_at_k:
                    num_retrieved_correct = 0
                    num_relevant_correct = 0
                    for query_id in query_relevant_docs:
                        query_law_id = query_id.split("@")[0]
                        for hit in top_hits[0:k_val]:
                            if hit['law_id'] == query_law_id:
                                num_retrieved_correct += 1
                        for hit in top_hits[0:k_val]:
                            if query_law_id == hit['law_id']:
                                num_relevant_correct += 1
                                break
                    precision = num_retrieved_correct / k_val
                    recall = num_relevant_correct / len(query_relevant_docs)
                    f2_score = self.calculate_f2(precision, recall)

                    precisions_at_k[k_val].append(precision)
                    recall_at_k[k_val].append(recall)
                    f2_score_at_k[k_val].append(f2_score)
                    
                # MRR@k
                for k_val in self.mrr_at_k:
                    for rank, hit in enumerate(top_hits[0:k_val]):
                        checked_hit = False
                        for query_id in query_relevant_docs:
                            query_law_id = query_id.split("@")[0]
                            if hit['law_id'] == query_law_id:
                                MRR[k_val] += 1.0 / (rank + 1)
                                checked_hit = True
                                break
                        if checked_hit:
                            break

                # NDCG@k
                for k_val in self.ndcg_at_k:
                    predicted_relevance = [1 if top_hit['law_id']+'@'+top_hit['article_id'] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]]
                    true_relevances = [1] * len(query_relevant_docs)

                    ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
                    ndcg[k_val].append(ndcg_value)

                # MAP@k
                for k_val in self.map_at_k:
                    num_correct = 0
                    sum_precisions = 0
                    for rank, hit in enumerate(top_hits[0:k_val]):
                        for query_id in query_relevant_docs:
                            query_law_id = query_id.split("@")[0]
                            if hit['law_id'] == query_law_id:
                                num_correct += 1
                                break
                        sum_precisions += num_correct / (rank + 1)

                    avg_precision = sum_precisions / k_val
                    AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)
        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])
        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])
        for k in f2_score_at_k:
            f2_score_at_k[k] = np.mean(f2_score_at_k[k])
        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])
        for k in MRR:
            MRR[k] /= len(self.queries)
        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        return {'accuracy@k': num_hits_at_k, 
                'precision@k': precisions_at_k,
                'recall@k': recall_at_k, 
                'f2-score@k': f2_score_at_k,
                'mrr@k': MRR, 
                'ndcg@k': ndcg, 
                'map@k': AveP_at_k}

    def output_scores(self, scores):

        print("\n-------- Accuracy ---------")
        for k in scores['accuracy@k']:
            print("Accuracy@{}: {:.2f}%".format(k, scores['accuracy@k'][k]*100))

        print("-------- Precision ---------")
        for k in scores['precision@k']:
            print("Precision@{}: {:.2f}%".format(k, scores['precision@k'][k]*100))

        print("-------- Recall ---------")
        for k in scores['recall@k']:
            print("Recall@{}: {:.2f}%".format(k, scores['recall@k'][k]*100))
        
        print("-------- F2 Score ---------")
        for k in scores['f2-score@k']:
            print("f2-score@{}: {:.2f}%".format(k, scores['f2-score@k'][k]*100))
        
        print("-------- MRR ---------")
        for k in scores['mrr@k']:
            print("MRR@{}: {:.4f}".format(k, scores['mrr@k'][k]))
        
        print("-------- NDCG ---------")
        for k in scores['ndcg@k']:
            print("NDCG@{}: {:.4f}".format(k, scores['ndcg@k'][k]))
        
        print("-------- Average Precision ---------")
        for k in scores['map@k']:
            print("MAP@{}: {:.4f}".format(k, scores['map@k'][k]))

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  #+2 as we start our idx at 0
        return dcg
    @staticmethod
    def calculate_f2(precision, recall):        
        return (5 * precision * recall) / (4 * precision + recall + 1e-20)

if __name__=="__main__":
    pass