import os
import logging
import argparse
from tqdm import tqdm
from utils import *
from vncorenlp import VnCoreNLP
from sklearn.model_selection import train_test_split

logging.basicConfig(
    format="%(asctime)s %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p %Z",
    level = logging.INFO
)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="./data", type=str, help="for loading question")
    parser.add_argument("--generated_data", default="./generated_data", type=str, help="path to paired data for contrastive learning")
    parser.add_argument("--word_segmenter", default="./VnCoreNLP/VnCoreNLP-1.1.1.jar", type=str)
    parser.add_argument("--train_ratio", default=0.65, type=float)
    args = parser.parse_args()

    train_question_answer = load_json(os.path.join(args.raw_data, "train_question_answer.json"))
    legal_dict  = load_json(os.path.join(args.generated_data, "legal_dict.json"))

    QA_train, QA_test = train_test_split(train_question_answer["items"], test_size=1-args.train_ratio, random_state=42)

    logging.info("Pairing the contrastive setences.")
    qa_pairs = [
        [clean_text(item["question"]), legal_dict[label["law_id"]+"@"+label["article_id"]]]\
            for item in QA_train for label in item["relevant_articles"]
    ]
    
    segmented_pairs = []
    logging.info("Load Word-Segmenter...")
    with VnCoreNLP(args.word_segmenter, 
    annotators="wseg,pos,ner,parse", 
    max_heap_size="-Xmx2g") as annotator:
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
    save_parameter(segmented_pairs, os.path.join(args.generated_data, "train_cl_pairs.pkl"))
    logging.info("Created training pairs successfully.")
    save_parameter(QA_test, os.path.join(args.generated_data, "test_cl_question_answer.pkl"))
    logging.info("Created test questions-answers.")