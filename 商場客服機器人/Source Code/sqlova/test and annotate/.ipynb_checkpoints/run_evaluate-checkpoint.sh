
#跑evaluate_ws.py
python3 evaluate_ws.py --pred_file --seed 1 --bS 4 --tepoch 1 --accumulate_gradients 2 --bert_type_abb uS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 

#跑train_loaddata.py
#python3 train_loaddata.py --do_train --seed 1 --bS 4 --tepoch 20 --accumulate_gradients 2 --bert_type_abb uS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --trained --do_infer

#跑train_getmodel.py #since a folder train is created
#jovyan@jupyter-xt121060:~/xt121-group11/sqlova$ cd ..
#jovyan@jupyter-xt121060:~/xt121-group11$ cd ~
#jovyan@jupyter-xt121060:~$ cd xt121-group11/sqlova/train
#python3 train_getmodel.py --do_train --seed 1 --bS 4 --tepoch 20 --accumulate_gradients 2 --bert_type_abb uS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --trained --do_infer



#參考資料https://blog.csdn.net/MirageTanker/article/details/127998036
#實際上的修改:助教建議將bS改小,從原本的--bS 16改成--bS 4
#查詢關鍵字:nvidia-smi,查詢關鍵字PYTORCH_CUDA_ALLOC_CONF,max_split_size_mb
#jovyan@jupyter-xt121060:~$ cd ..
#jovyan@jupyter-xt121060:/home$ cd ~
#jovyan@jupyter-xt121060:~$ cd xt121-group11/sqlova
#jovyan@jupyter-xt121060:~/xt121-group11/sqlova$ bash run_train.sh

"""
In the prompt "python3 train.py --do_train --seed 1 --bS 4 --tepoch 20 --accumulate_gradients 2 --bert_type_abb uS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --trained --do_infer", the "--do_train" option specifies that the model should be trained, while the "--trained" option indicates that a pre-trained model should be used for training.

Specifically, the "--do_train" option is used to train the model, and it triggers the training mode of the script. When this option is used, the script reads in a training dataset and trains the model on that dataset.

On the other hand, the "--trained" option is used to specify that a pre-trained model should be used for training instead of initializing the model from scratch. This can be useful when you want to fine-tune a pre-trained model on a specific task or dataset.

"""

"""
`parser` in this context is an instance of the `argparse.ArgumentParser` class, which is a Python library for parsing command-line arguments and options.

`construct_hyper_param` is a function that takes an instance of the `argparse.ArgumentParser` class as an argument, and adds several arguments to it using the `add_argument()` method. In this case, it adds the following arguments:

- `--do_train`: a boolean flag indicating whether to run the training script.
- `--do_infer`: a boolean flag indicating whether to run the inference script.
- `--infer_loop`: a boolean flag indicating whether to run the inference script in a loop. 

These arguments can be used to control the behavior of the script when it is run from the command line. For example, if the `--do_train` flag is set, the script will run the training code, and if the `--do_infer` flag is set, it will run the inference code.


"""