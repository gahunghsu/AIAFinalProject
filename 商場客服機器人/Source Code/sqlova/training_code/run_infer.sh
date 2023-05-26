#original
#python3 train.py --do_infer --infer_loop --trained --bert_type_abb uL --max_seq_length 222

python3 train.py --do_infer --infer_loop --bert_type_abb uL --max_seq_length 222


#The --infer_loop option in the command you provided is used to specify whether or not to run the inference in a loop. When this #option is present, the program will repeatedly prompt the user for input sto generate SQL queries until the user enters the string #"exit". This can be useful when trying out the model on a specific dataset or in an interactive setting, where the user may want #to input multiple questions and see the corresponding SQL queries. If --infer_loop is not present, the program will simply run #inference on the input provided in the --input_file option (if specified) and output the corresponding SQL queries.


