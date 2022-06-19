import os
import logging
import argparse
from torch import positive
from tqdm import tqdm
from utils import *
from vncorenlp import VnCoreNLP
from rank_bm25 import BM25Okapi
from sklearn.model_selection import train_test_split

logging.basicConfig(
    format="%(asctime)s %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p %Z",
    level = logging.INFO
)
class Config:
    save_bm25 = "./saved_model"
    top_k_bm25 = 10
    bm25_k1 = 0.4
    bm25_b = 0.6

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="./data", type=str, help="for loading question")
    parser.add_argument("--saved_model", default="./saved_model", type=str, help="path to saved bm25 model")
    parser.add_argument("--model_name", default="", type=str, help="bm25 model name")
    parser.add_argument("--legal_data", default="./generated_data", type=str, help="path to paired data for contrastive learning")
    parser.add_argument("--word_segmenter", default="./VnCoreNLP/VnCoreNLP-1.1.1.jar", type=str)
    parser.add_argument("--train_ratio", default=0.65, type=float)
    args = parser.parse_args()
    config = Config()

    legal_dict = load_json(os.path.join(args.legal_data, "legal_dict.json"))
    idx2lid = {idx: lid for idx, lid in enumerate(legal_dict)}

    train_question_answer = load_json(os.path.join(args.raw_data, "train_question_answer.json"))
    QA_train, QA_dev = train_test_split(train_question_answer["items"], test_size=1-args.train_ratio, random_state=42)

    if os.path.exists(os.path.join(args.saved_model, args.model_name)):
        logging.info("Train BM25 model")
        seg_doc_refers = load_parameter(os.path.join(args.legal_data, "segmented_docs.pkl"))
        bm25 = BM25Okapi(seg_doc_refers)
    else:
        logging.info("Load Bm25 Model.")
        bm25 = load_parameter(os.path.join(args.saved_model, args.model_name))
    
    logging.info("Pairing the contrastive samples.")
    triple_pairs = []

    with VnCoreNLP(args.word_segmenter, annotators="wseg,pos,ner,parse", max_heap_size="-Xmx2g") as annotator:
        for item in tqdm(QA_train, desc="processing"):
            
            question = clean_text(item["question"])
            tokenized_question = flatten2DList(annotator.tokenize(question))
            segmented_question = " ".join(tokenized_question)

            doc_scores = bm25.get_scores(tokenized_question)
            topk_min_ids = (-doc_scores).argsort()[-config.top_k_bm25:]
            candidates = [idx2lid[idx] for idx in topk_min_ids]
            ground_truths = [label["law_id"]+"@"+label["article_id"] for label in item["relevant_articles"]]
            for lid in ground_truths:
                if lid in candidates:
                    candidates.pop(lid)
                positive = legal_dict[lid]
                negative = legal_dict[candidates[-1]]

                seg_positive = " ".join(flatten2DList(annotator.tokenize(positive)))
                seg_negative = " ".join(flatten2DList(annotator.tokenize(negative)))
                
                triple_pairs.append(
                    [segmented_question, seg_positive, seg_negative]
                )
        
    save_parameter(triple_pairs, os.path.join(args.legal_data, "train_triple_pairs.pkl"))
    save_parameter(QA_dev, os.path.join(args.legal_data, "dev_triple_pairs.pkl"))
