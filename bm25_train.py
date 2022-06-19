import os
import logging
import argparse
from tqdm import tqdm
from utils import *
from vncorenlp import VnCoreNLP
from rank_bm25 import BM25Okapi

logging.basicConfig(
    format="%(asctime)s %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p %Z",
    level = logging.INFO
)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="./raw_data", type=str, help="path to question answer set")
    parser.add_argument("--legal_data", default="./generated_data", type=str, help="path to paired data for contrastive learning")
    parser.add_argument("--model_name", default="bm25_model.pkl", type=str)
    parser.add_argument("--word_segmenter", default="./VnCoreNLP/VnCoreNLP-1.1.1.jar", type=str)
    parser.add_argument("--train_ratio", default=0.65, type=float)
    args = parser.parse_args()

    
    doc_refers = load_parameter(os.path.join(args.legal_data, "doc_refers_saved.pkl"))


    logging.info("Load Word-Segmenter...")
    segmented_docs = []
    with VnCoreNLP(
        args.word_segmenter, 
        annotators="wseg,pos,ner,parse", 
        max_heap_size="-Xmx2g"
    ) as annotator:
        for doc in tqdm(doc_refers, desc="processing"):
            try:
                seg_doc = flatten2DList(annotator.tokenize(doc))
            except:
                seg_doc = doc.split()
            segmented_docs.append(seg_doc)

    bm25 = BM25Okapi(segmented_docs)
    save_parameter(bm25, os.path.join(args.saved_model, args.model_name))
    