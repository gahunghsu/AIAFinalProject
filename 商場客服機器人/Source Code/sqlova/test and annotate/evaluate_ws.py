#!/usr/bin/env python
import json
from argparse import ArgumentParser
from tqdm import tqdm
from wikisql.lib.dbengine import DBEngine
from wikisql.lib.query import Query
from wikisql.lib.common import count_lines

import os

# Jan1 2019. Wonseok. Path info has added to original wikisql/evaluation.py
# Only need to add "query" (essentially "sql" in original data) and "table_id" while constructing file.

if __name__ == '__main__':

    # Hyper parameters
    mode = 'test'
    ordered = False

    dset_name = 'wikisql_tok'
    saved_epoch = 'best'  # 30-162

    # Set path
    path_h = './' # change to your home folder
    path_wikisql_tok = os.path.join(path_h, 'data', 'wikisql_tok')
    path_save_analysis = '.'
    

    # Path for evaluation results.
    #./data/WikiSQL-1.1/data( path_wikisql0的路徑)
    path_wikisql0 = os.path.join(path_h,'data/WikiSQL-1.1/data') 
    
    #./data/WikiSQL-1.1/data/dev.jsonl
    path_source = os.path.join(path_wikisql0, f'{mode}.jsonl')
    
    #./data/WikiSQL-1.1/data/dev.db
    path_db = os.path.join(path_wikisql0, f'{mode}.db')
    
    #.results_dev.jsonl
    path_pred = os.path.join(path_save_analysis, f'results_{mode}.jsonl') #原本的
    #path_pred = os.path.join(path_wikisql0, f'results_{mode}.jsonl') 可以考慮看看是否要用這個
    
    


    # For the case when use "argument"
    parser = ArgumentParser()
    parser.add_argument('--source_file', help='source file for the prediction', default=path_source) #./data/WikiSQL-1.1/data/dev.jsonl
    parser.add_argument('--db_file', help='source database for the prediction', default=path_db) #./data/WikiSQL-1.1/data/dev.db
    parser.add_argument('--pred_file', help='predictions by the model', default=path_pred)#.results_dev.jsonl
    parser.add_argument('--ordered', action='store_true', help='whether the exact match should consider the order of conditions')
    args = parser.parse_args()
    args.ordered=ordered

    engine = DBEngine(args.db_file)
    exact_match = []
    with open(args.source_file) as fs, open(args.pred_file) as fp:
        grades = []
        for ls, lp in tqdm(zip(fs, fp), total=count_lines(args.source_file)):
            eg = json.loads(ls)
            ep = json.loads(lp)
            qg = Query.from_dict(eg['sql'], ordered=args.ordered)
            gold = engine.execute_query(eg['table_id'], qg, lower=True)
            pred = ep.get('error', None)
            qp = None
            if not ep.get('error', None):  #def from_dict(cls, d, ordered=False):
                                                   #return cls(sel_index=d['sel'], agg_index=d['agg'],
                                                    #conditions=d['conds'], ordered=ordered)
                try:
                    qp = Query.from_dict(ep['query'], ordered=args.ordered)
                    pred = engine.execute_query(eg['table_id'], qp, lower=True)
                except Exception as e:
                    pred = repr(e)
            correct = pred == gold
            match = qp == qg
            grades.append(correct)
            exact_match.append(match)

        print(json.dumps({
            'ex_accuracy': sum(grades) / len(grades),
            'lf_accuracy': sum(exact_match) / len(exact_match),
            }, indent=2))


