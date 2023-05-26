#跑第一次predict(terminal 5)(for predict dev)
python3 predict.py --model_file model_best.pt --bert_model_file model_bert_best.pt --bert_path ./data_and_model/ --data_path ./gaga_0516 --split customer --result_path ./gaga_0516


#參考資料https://blog.csdn.net/MirageTanker/article/details/127998036
#實際上的修改:助教建議將bS改小,從原本的--bS 16改成--bS 4
#查詢關鍵字:nvidia-smi,查詢關鍵字PYTORCH_CUDA_ALLOC_CONF,max_split_size_mb
#jovyan@jupyter-xt121060:~$ cd ..
#jovyan@jupyter-xt121060:/home$ cd ~
#jovyan@jupyter-xt121060:~$ cd xt121-group11/sqlova
#jovyan@jupyter-xt121060:~/xt121-group11/sqlova$ bash run_train.sh

"""
# Use existing model to predict sql from tables and questions.
#
# For example, you can get a pretrained model from https://github.com/naver/sqlova/releases:
#    https://github.com/naver/sqlova/releases/download/SQLova-parameters/model_bert_best.pt
#    https://github.com/naver/sqlova/releases/download/SQLova-parameters/model_best.pt
#
# Make sure you also have the following support files (see README for where to get them):
#    - bert_config_uncased_*.json
#    - vocab_uncased_*.txt
#
# Finally, you need some data - some files called:
#    - <split>.db
#    - <split>.jsonl
#    - <split>.tables.jsonl
#    - <split>_tok.jsonl         # derived using annotate_ws.py
# You can play with the existing train/dev/test splits, or make your own with
# the add_csv.py and add_question.py utilities.
#
# Once you have all that, you are ready to predict, using:
#   python predict.py \
#     --bert_type_abb uL \       # need to match the architecture of the model you are using
#     --model_file <path to models>/model_best.pt            \
#     --bert_model_file <path to models>/model_bert_best.pt  \
#     --bert_path <path to bert_config/vocab>  \
#     --result_path <where to place results>                 \
#     --data_path <path to db/jsonl/tables.jsonl>            \
#     --split <split>
#
# Results will be in a file called results_<split>.jsonl in the result_path.

"""

