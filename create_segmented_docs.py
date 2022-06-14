import os
import argparse
from tqdm import tqdm
from utils import *
from vncorenlp import VnCoreNLP


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="./data", type=str, help="for loading question")
    parser.add_argument("--legal_data", default="./generated_data", type=str, help="path to doc for reference")
    parser.add_argument("--word_segmentator", default="./VnCoreNLP/VnCoreNLP-1.1.1.jar", type=str, help="path to word segmenter")
    args = parser.parse_args()

    doc_refers = load_parameter(os.path.join(args.legal_data, "doc_refers_saved.pkl"))

    print(">> Load Word-Segmenter...")
    annotator = VnCoreNLP(
        args.word_segmentator, 
        annotators="wseg,pos,ner,parse", 
        max_heap_size="-Xmx2g"
    )
    segmented_docs = []
    for doc in tqdm(doc_refers, desc="processing"):
        try:
            seg_doc = " ".join(flatten2DList(annotator.tokenize(doc)))
        except:
            seg_doc = doc
        segmented_docs.append(seg_doc)

    save_parameter(segmented_docs, os.path.join(args.legal_data, "segmented_docs.pkl"))
    print("Created segmented docs for reference successfully.")
    
