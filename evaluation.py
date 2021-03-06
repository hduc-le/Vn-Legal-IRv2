import os
import logging
logging.basicConfig(
    format="%(asctime)s %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p %Z",
    level = logging.INFO
)
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
    parser.add_argument("--generated_data", default="generated_data", type=str, help="path to save doc refer.")
    parser.add_argument("--model_name_or_path", default="vinai/bartpho-word", type=str, help="path to pretrained model")
    parser.add_argument("--tokenizer_name_or_path", default="vinai/bartpho-word", type=str, help="path to pretrained tokenizer")
    parser.add_argument("--word_segmenter", default="./VnCoreNLP/VnCoreNLP-1.1.1.jar", type=str, help="path to word segmenter")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size for embedding legal docs")
    parser.add_argument("--max_seq_len", default=300, type=int)
    parser.add_argument("--save_to", default=None, type=str, help="path to save evaluation results")
    parser.add_argument("--name", default="evaluation_results.csv", type=str, help="csv name file for evaluation results")
    args = parser.parse_args()

    device = get_device()
    
    logging.info("Prepare data for evaluation")
    doc_refers = load_parameter(os.path.join(args.generated_data, "doc_refers_saved.pkl"))
    legal_dict = load_json(os.path.join(args.generated_data, "legal_dict.json"))

    logging.info("Load Word-Segmenter...")
    annotator = VnCoreNLP(
        args.word_segmenter, 
        annotators="wseg,pos,ner,parse", 
        max_heap_size="-Xmx2g"
    )

    logging.info("Load model...")
    model = AutoModel.from_pretrained(args.model_name_or_path)
    
    logging.info("Load pretrained tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    

    evaluator = Evaluator(model, tokenizer, annotator)
    logging.info("Encode corpus:")


    corpus_embeddings = evaluator.encode(
        doc_refers,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        show_progress_bar=True,
        convert_to_tensor=True
    )


    logging.info("Corpus has been encoded successfully")
    logging.info("Loading Queries:")
    qa_test = load_parameter(os.path.join(args.generated_data, "test_cl_question_answer.pkl"))
    
    dev_queries = {}
    relevant_docs = {}
    for item in tqdm(qa_test, desc="Loading"):
        dev_queries[item["question_id"]] = item["question"]
        relevant_docs[item["question_id"]] = set([doc["law_id"] + "@" + doc["article_id"] for doc in item["relevant_articles"]])
    
    ## Run evaluator
    logging.info("Queries: {}".format(len(dev_queries)))
    logging.info("Corpus: {}".format(len(legal_dict)))

    logging.info("Start Evaluation")
    ir_evaluator = InformationRetrievalEvaluator(queries=dev_queries, 
                                                corpus=legal_dict, 
                                                relevant_docs=relevant_docs,
                                                show_progress_bar=True,
                                                corpus_chunk_size=1000,
                                                mrr_at_k=[1,3,5,10],
                                                ndcg_at_k=[1,3,5,10],
                                                name=args.name)

    ir_evaluator(model=evaluator, 
        output_path=args.save_to, 
        corpus_embeddings=corpus_embeddings, 
        max_seq_len=args.max_seq_len
    )
    
    # close the sever
    annotator.close()