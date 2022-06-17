import logging
import os
import argparse
from utils import load_parameter, save_parameter
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    format="%(asctime)s %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p %Z",
    level = logging.INFO
)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--legal_data", 
                        default="./generated_data", 
                        type=str, 
                        help="path to input data")
    parser.add_argument("--max_features", 
                        default=300, 
                        type=int, 
                        help="number of tf-idf features' dimension")
    parser.add_argument("--save_path",
                        default="../saved_model",
                        type=str,
                        help="the path/location for saving model"
    )
    args = parser.parse_args()
    
    doc_refers = load_parameter(os.path.join(args.legal_data, "doc_refers_saved.pkl"))
    segmented_docs = load_parameter(os.path.join(args.legal_data, "segmented_docs.pkl"))

    save_format = f"tfidf-model-{args.max_features}.pkl"
    logging.info("[W] Learning Tfidf Vectorizer ...")
    tfidf_vectorizer = TfidfVectorizer(max_features=args.max_features)
    tfidf_vectorizer.fit(segmented_docs)

    save_parameter(tfidf_vectorizer, os.path.join(args.save_path, save_format))