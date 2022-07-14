import os
import logging
import argparse
from tqdm import tqdm
from train_memory import Q_memories
from utils import *
from vncorenlp import VnCoreNLP
from rank_bm25 import BM25Okapi
from sklearn.model_selection import train_test_split

logging.basicConfig(
    format="%(asctime)s %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p %Z",
    level = logging.INFO
)

def make_pair_question_answer(QA_set, annotator, legal_dict):
    pairs = []
    for item in tqdm(QA_set):
        try:
            seg_question = " ".join(flatten2DList(annotator.tokenize(clean_text(item["question"]))))
            for label in item["relevant_articles"]:
                pos_sent = legal_dict[label["law_id"]+"@"+label["article_id"]]
                seg_pos_sent = " ".join(flatten2DList(annotator.tokenize(pos_sent)))
                pairs.append(
                    [seg_question, seg_pos_sent]
                )
        except:
            pass
    return pairs

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="./raw_data", type=str, help="path to question answer set")
    parser.add_argument("--generated_data", default="./generated_data", type=str, help="path to paired data for contrastive learning")
    parser.add_argument("--model_name", default="bm25_model.pkl", type=str)
    parser.add_argument("--word_segmenter", default="./VnCoreNLP/VnCoreNLP-1.1.1.jar", type=str)
    parser.add_argument("--saved_model", default="./saved_model", type=str)
    args = parser.parse_args()

    legal_dict = load_json(os.path.join(args.generated_data, "legal_dict.json"))
    train_question_answer = load_json(os.path.join(args.raw_data, "train_question_answer.json"))

    Q_train, Q_dev = train_test_split(train_question_answer["items"], test_size=0.7, random_state=42)
    Q_new, Q_test = train_test_split(Q_dev, test_size=0.4, random_state=42)
    

    annotator = VnCoreNLP(
        "./VnCoreNLP/VnCoreNLP-1.1.1.jar", 
        annotators="wseg,pos,ner,parse", 
        max_heap_size="-Xmx2g"
    )

    Q_memories = make_pair_question_answer(Q_train, annotator=annotator, legal_dict=legal_dict)
    Q_new = make_pair_question_answer(Q_new, annotator=annotator, legal_dict=legal_dict)

    tokenized_docs = [sent.split() for sent, _ in Q_memories]
    bm25 = BM25Okapi(tokenized_docs)
    
    
    save_parameter(bm25, os.path.join(args.saved_model, "bm25_model.pkl"))
    save_parameter(Q_new, os.path.join(args.generated_data, "Q_new.pkl"))
    save_parameter(Q_memories, os.path.join(args.generated_data, "Q_memories.pkl"))
    save_parameter(Q_test, os.path.join(args.generated_data, "Q_test.pkl"))

    annotator.close()