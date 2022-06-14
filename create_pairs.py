import os
import argparse
from tqdm import tqdm
from utils import *
from vncorenlp import VnCoreNLP
from sklearn.model_selection import train_test_split


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="./data", type=str, help="for loading question")
    parser.add_argument("--legal_data", default="./generated_data", type=str, help="path to paired data for contrastive learning")
    parser.add_argument("--train_ratio", default=0.65, type=float)
    args = parser.parse_args()

    train_question_answer = load_json(os.path.join(args.raw_data, "train_question_answer.json"))
    legal_dict  = load_json(os.path.join(args.legal_data, "legal_dict.json"))

    qa_train, qa_test = train_test_split(train_question_answer["items"], test_size=1-args.train_ratio, random_state=42)

    print(">> Pairing the contrastive setences.")
    qa_pairs = [
        [clean_text(item["question"]), legal_dict[label["law_id"]+"@"+label["article_id"]]]\
            for item in qa_train for label in item["relevant_articles"]
    ]
    
    print(">> Load Word-Segmenter...")
    annotator = VnCoreNLP(
        args.word_segmentator, 
        annotators="wseg,pos,ner,parse", 
        max_heap_size="-Xmx2g"
    )
    segmented_pairs = []
    for pair in tqdm(qa_pairs, desc="Pairing"):
        try:
            sent0 = " ".join(flatten2DList(annotator.tokenize(pair[0])))
            sent1 = " ".join(flatten2DList(annotator.tokenize(pair[1])))
        except:
            sent0 = pair[0]
            sent1 = pair[1]
        segmented_pairs.append(
            [sent0, sent1]
        )
    save_parameter(segmented_pairs, os.path.join(args.legal_data, "train_pairs.pkl"))
    print("Created training pairs successfully.")
    save_parameter(qa_test, os.path.join(args.legal_data, "test_question_answer.pkl"))
    print("Created test questions-answers.")