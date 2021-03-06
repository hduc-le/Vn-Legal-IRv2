import os 
import argparse
import logging

from tqdm import tqdm
from utils import *
logging.basicConfig(
    format="%(asctime)s %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p %Z",
    level = logging.INFO
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="data", type=str, help="path to raw data")
    parser.add_argument("--generated_data", default="generated_data", type=str, help="path to save doc refer.")
    args = parser.parse_args()

    data = load_json(os.path.join(args.raw_data, "legal_corpus.json"))

    
    logging.info("Start create legal dict.")

    legal_dict = {
        doc["law_id"]+"@"+article["article_id"]: clean_text(article["title"] + " " + article["text"]) \
            for doc in tqdm(data, desc="Creating legal dict") \
                for article in doc["articles"]
    }
    
    logging.info("Start create doc refer.")
    doc_refers = [doc for doc in legal_dict.values()]
    
    os.makedirs(args.generated_data, exist_ok=True)
    save_parameter(doc_refers, os.path.join(args.generated_data, "doc_refers_saved.pkl"))
    save_json(legal_dict, os.path.join(args.generated_data, "legal_dict.json"))

    
    logging.info("Created Doc Data.")
    logging.info("Created legal dictionary.")
    
    