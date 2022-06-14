import os 
import argparse
from tqdm import tqdm
from utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="data", type=str, help="path to raw data")
    parser.add_argument("--legal_data", default="generated_data", type=str, help="path to save doc refer.")
    args = parser.parse_args()

    data = load_json(os.path.join(args.raw_data, "legal_corpus.json"))

    print("=====================") 
    print("Start create legal dict.")

    legal_dict = {
        doc["law_id"]+"@"+article["article_id"]: clean_text(article["title"] + " " + article["text"]) \
            for doc in tqdm(data, desc="Creating legal dict") \
                for article in doc["articles"]
    }
    print("=====================") 
    print("Start create doc refer.")
    doc_refers = [doc for doc in legal_dict.values()]
    
    os.makedirs(args.legal_data, exist_ok=True)
    save_parameter(doc_refers, os.path.join(args.legal_data, "doc_refers_saved.pkl"))
    save_json(legal_dict, os.path.join(args.legal_data, "legal_dict.json"))

    print("=====================") 
    print("Created Doc Data.")
    print("Created legal dictionary.")
    
    