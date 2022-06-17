import os
import logging
logging.basicConfig(
    format="%(asctime)s %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p %Z"
)
import pandas as pd 
import argparse
import warnings
warnings.filterwarnings('ignore')

from vncorenlp import VnCoreNLP
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from models.ModelEvaluator import Evaluator
from models.InformationRetrievalEvaluator import InformationRetrievalEvaluator
from utils import *
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--legal_data", default="generated_data", type=str, help="path to save doc refer.")
    parser.add_argument("--model_name_or_path", default="vinai/bartpho-mlm", type=str, help="path to pretrained model")
    parser.add_argument("--tokenizer_name_or_path", default="vinai/bartpho-mlm", type=str, help="path to pretrained tokenizer")
    parser.add_argument("--word_segmenter", default="./VnCoreNLP/VnCoreNLP-1.1.1.jar", type=str, help="path to word segmenter")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size for embedding legal docs")
    # parser.add_argument("--model_type", default="hg", type=str, help="set `hg` if model for evaluation is inherited from PreTrainedModel base class, `pt` if it's a pytorch custom model")
    # parser.add_argument("--eval_mode", default="full_id", type=str, help="the precision of evaluation, options are `full_id` and `law_id`")
    parser.add_argument("--save_to", default=None, type=str, help="path to save evaluation results")
    parser.add_argument("--name", default="evaluation_results.csv", type=str, help="csv name file for evaluation results")
    args = parser.parse_args()

    device = get_device()
    logging.info(">> Prepare data for evaluation")
    doc_refers = load_parameter(os.path.join(args.legal_data, "doc_refers_saved.pkl"))
    legal_dict = load_json(os.path.join(args.legal_data, "legal_dict.json"))

    logging.info(">> Load Word-Segmenter...")
    annotator = VnCoreNLP(
        args.word_segmenter, 
        annotators="wseg,pos,ner,parse", 
        max_heap_size="-Xmx2g"
    )
    if args.model_type == "hg":
        logging.info(">> Load model...")
        model = AutoModel.from_pretrained(args.model_name_or_path)
        
        logging.info(">> Load pretrained tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    else:
        raise NotImplementedError("Still not Implement !!!")

    evaluator = Evaluator(model, tokenizer, annotator)
    
    logging.info(">> Loading Queries:")
    qa_test = load_parameter(os.path.join(args.legal_data, "test_question_answer.pkl"))
    dev_queries = {}
    relevant_docs = {}
    for item in tqdm(qa_test, desc="Loading"):
        dev_queries[item["question_id"]] = item["question"]
        relevant_docs[item["question_id"]] = set([doc["law_id"] + "@" + doc["article_id"] for doc in item["relevant_articles"]])
    
    ## Run evaluator
    logging.info("Queries: {}".format(len(dev_queries)))
    logging.info("Corpus: {}".format(len(legal_dict)))

    ir_evaluator = InformationRetrievalEvaluator(dev_queries, legal_dict, relevant_docs,
                                                show_progress_bar=True,
                                                corpus_chunk_size=args.batch_size,
                                                mrr_at_k=[1,3,5,10],
                                                ndcg_at_k=[1,3,5,10],
                                                name=args.name)

    ir_evaluator(evaluator, output_path=args.save_to)

    # print("============== Start Evaluation ==============")
    # ir_evaluator = InformationRetrievalEvaluator(queries=queries, 
    #                                       corpus=legal_dict, 
    #                                       relevant_docs=relevant_docs, 
    #                                       corpus_chunk_size=1000, 
    #                                       show_progress_bar=True, 
    #                                       eval_mode=args.eval_mode)

    # scores, queries_result_list = ir_evaluator.compute_metrices(model=evaluator, 
    #                                                         corpus_embeddings=corpus_embeddings)
    