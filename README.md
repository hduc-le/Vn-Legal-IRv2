# Vn-Legal-IRv2

# Training
## Preprocessing
```python3
python preprocessing.py --raw_data="./raw_data" \
                     --generated_data="./generated_data"\
```                          
### Train Query-Transformation Model
Execute the following command to train the bm25 model for creating the memory of queries.
```python3
python bm25_train.py --raw_data="./raw_data"\
                     --generated_data="./generated_data"\
                     --model_name="./bm25_model.pkl"\
                     --word_segmenter="./VnCoreNLP/VnCoreNLP-1.1.1.jar"\
                     --saved_model="./saved_model"\
```
Run below command to start training memory model:
```python3
python train_memory.py --raw_data="./raw_data"\
                       --generated_data="./generated_data"\
                       --model_name_or_path="vinai/bartpho-word"\
                       --saved_model="saved_model/model-memory.pth"\
                       --max_seq_len=300\
                       --learning_rate=0.2\
```

### Train Siamese-Model using Contrastive Learning
```python3
python create_pairs.py --raw_data="./raw_data"\
                       --generated_data="./generated_data"\
                       --word_segmenter="./VnCoreNLP/VnCoreNLP-1.1.1.jar"\
                       --train_ratio=0.65\
```
```python3
python train_cl.py --generated_data="./generated_data"\
                   --model_name_or_path="viai/bartpho-word"\
                   --saved_model="./saved_model"\
                   --max_seq_len=300\
                   --temperature=0.1\
                   --learning_rate=5e-5\
                   --lr_decay=True\
                   --decay_rate=0.96\
                   --batch_size=4\
                   --num_epochs=5\
                   
```
