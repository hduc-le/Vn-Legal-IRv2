import os
import argparse
import warnings
warnings.filterwarnings('ignore')

from vncorenlp import VnCoreNLP
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel
from models.ModelEvaluator import Evaluator
from models.InformationRetrievalEvaluator import InformationRetrievalEvaluator
from utils import *
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--legal_data", default="generated_data", type=str, help="path to save doc refer.")
    parser.add_argument("--model_name_or_path", default="vinai/bartpho-mlm", type=str, help="path to pretrained model")
    parser.add_argument("--tokenizer_name_or_path", default="vinai/bartpho-mlm", type=str, help="path to pretrained tokenizer")
    parser.add_argument("--word_segmenter", default="./VnCoreNLP/VnCoreNLP-1.1.1.jar", type=str, help="path to word segmenter")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size for embedding legal docs")
    parser.add_argument("--pooling_output", default=True, type=bool, help="return model output with pooling operator")
    parser.add_argument("--eval_mode", default="full_id", type=str, help="the precision of evaluation, options are `full_id` and `law_id`")
    args = parser.parse_args()

    device = get_device()
    print(">> Prepare data for evaluation")
    doc_refers = load_parameter(os.path.join(args.legal_data, "doc_refers_saved.pkl"))
    legal_dict = load_json(os.path.join(args.legal_data, "legal_dict.json"))

    print(">> Load Word-Segmenter...")
    annotator = VnCoreNLP(
        args.word_segmenter, 
        annotators="wseg,pos,ner,parse", 
        max_heap_size="-Xmx2g"
    )
    print(">> Load pretrained model...")
    model = AutoModel.from_pretrained(args.model_name_or_path)

    print(">> Load pretrained tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    evaluator = Evaluator(model, tokenizer, annotator)
    print(">> Encode legal docs for reference")
    corpus_embeddings = evaluator.encode(
            doc_refers, 
            batch_size=args.batch_size, 
            convert_to_tensor=True, 
            device=device, 
            show_progress_bar=True,
            encoded_type="pooler_output" if args.pooling_output else "last_hidden"
        )
    print(">> Load Queries")
    qa_test = load_parameter(os.path.join(args.legal_data, "test_question_answer.pkl"))
    queries = {}
    relevant_docs = {}
    for item in tqdm(qa_test, desc="Loading"):
        queries[item["question_id"]] = item["question"]
        relevant_docs[item["question_id"]] = set([doc["law_id"] + "@" + doc["article_id"] for doc in item["relevant_articles"]])
    print(f"Number of queries: {len(queries)}")
    

    print("============== Start Evaluation ==============")
    ir_evaluator = InformationRetrievalEvaluator(queries=queries, 
                                          corpus=legal_dict, 
                                          relevant_docs=relevant_docs, 
                                          corpus_chunk_size=1000, 
                                          show_progress_bar=True, 
                                          eval_mode=args.eval_mode)

    scores, queries_result_list = ir_evaluator.compute_metrices(model=evaluator, 
                                                            corpus_embeddings=corpus_embeddings)