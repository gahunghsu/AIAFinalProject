#!/usr/bin/env python

# Add a line of json representing a question into <split>.jsonl
# Call as:
#   python add_question.py <split> <table id> <question>
#
# This utility is not intended for use during training.  A dummy label is added to the
# question to make it loadable by existing code.
#
# For example, suppose we downloaded this list of us state abbreviations:
#   https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/USstateAbbreviations.csv
# Let's rename it as something short, say "abbrev.csv"
# Now we can add it to a split called say "playground":
#   python add_csv.py playground abbrev.csv
"""
In the context of the WikiSQL dataset, a "split" refers to a specific subset of the data that is used for a particular purpose, such as training or evaluation. For example, the dataset may be divided into a training split, a validation split, and a test split. In this case, the phrase "add it to a split called say 'playground'" means to add the data from the CSV file into a specific subset of the dataset that has been named "playground". The purpose of this split may vary depending on the needs of the user, but it could be used for testing out code or experimenting with different models.
"""

# And now we can add a question about it to the same split:
#   python add_question.py playground abbrev "what state has ansi digits of 11"
# The next step would be to annotate the split:
#   python annotate_ws.py --din $PWD --dout $PWD --split playground
# Then we're ready to run prediction on the split with predict.py

"""
This is a Python script that takes in command line arguments and adds a question to a JSON file for use with the WikiSQL dataset. Here's how it works:

1. The script defines a function `question_to_json` that takes in a table ID, a question, and a file name for the JSON file.
2. Inside the `question_to_json` function, a new record is created as a dictionary with a `phase` field of 1 (this is just a dummy value), the specified table ID, the specified question, and an empty SQL query.
3. The `record` dictionary is written to the JSON file using the `json.dump` function.
4. The script parses command line arguments using the `argparse` module. It expects three arguments: the name of the split to add the question to, the table ID to associate with the question, and the text of the question itself.
5. The script calls `question_to_json` with the specified table ID, question, and file name (which is constructed from the split name).
6. Finally, the script prints a message indicating that the question has been added to the JSON file.

To use this script, you would run it from the command line like this:

```
python add_question.py <split> <table id> <question>
```

Here's an example command:

```
python add_question.py playground abbrev "what state has ansi digits of 11"
```

This would add a new question to a split called "playground" with a table ID of "abbrev" and the specified text. Note that the question will have a dummy label, so it won't be usable for training until you annotate the split with the `annotate_ws.py` script.


"""


import argparse, csv, json

from sqlalchemy import Column, create_engine, Integer, MetaData, String, Table
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import create_session, mapper

def question_to_json(table_id, question, json_file_name):
    record = {
        'phase': 1,
        'table_id': table_id,
        'question': question,
        'sql': {'sel': 0, 'conds': [], 'agg': 0}
    }
    with open(json_file_name, 'a+') as fout:
        json.dump(record, fout)
        fout.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('table_id')
    parser.add_argument('question', type=str, nargs='+')
    args = parser.parse_args()
    json_file_name = '{}.jsonl'.format(args.split)
    question_to_json(args.table_id, " ".join(args.question), json_file_name)
    print("Added question (with dummy label) to {}".format(json_file_name))
