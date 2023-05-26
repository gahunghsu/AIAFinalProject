#!/usr/bin/env python

# Add a CSV file as a table into <split>.db and <split>.tables.jsonl
# Call as:
#   python add_csv.py <split> <filename.csv>
# For a CSV file called data.csv, the table will be called table_data in the .db
# file, and will be assigned the id 'data'.
# All columns are treated as text - no attempt is made to sniff the type of value
# stored in the column.

"""
a Python script that performs several functions related to processing CSV files and converting them to SQLite databases and JSON files. 

"""

import argparse, csv, json, os, re
from sqlalchemy import Column, create_engine, MetaData, String, Table

#The function takes a table ID and returns a string representing the table name.
def get_table_name(table_id):
    return 'table_{}'.format(table_id)

#The function csv_to_sqlite() converts a CSV file to an SQLite database. It takes the table ID, CSV file name, SQLite #file name, and an optional working folder as input. It uses the create_engine function from SQLAlchemy to create an #SQLite engine. Then, it opens the CSV file, reads its contents using csv.DictReader, creates a table in the SQLite #database based on the CSV columns using the Table class, and inserts the data from the CSV file into the SQLite table.
def csv_to_sqlite(table_id, csv_file_name, sqlite_file_name, working_folder='.'):
    sqlite_file_name = os.path.join(working_folder, sqlite_file_name)
    csv_file_name = os.path.join(working_folder, csv_file_name)
    #initializes an SQLAlchemy engine object that can be used to interact with the SQLite database specified by the #provided file name.
    engine = create_engine('sqlite:///{}'.format(sqlite_file_name))

    with open(csv_file_name) as f:
        metadata = MetaData(bind=engine)
        cf = csv.DictReader(f, delimiter=',')
        simple_name = dict([(name, 'col%d' % i) for i, name in enumerate(cf.fieldnames)])
        table = Table(get_table_name(table_id), metadata,
                      *(Column(simple_name[name], String())
                        for name in cf.fieldnames))
        table.drop(checkfirst=True)
        table.create()
        for row in cf:
            row = dict((simple_name[name], val) for name, val in row.items())
            table.insert().values(**row).execute()
    return engine


def is_num(val):
    pattern = re.compile(r'[-+]?\d*\.\d+|\d+')
    if pattern.match(val):
        return True
    else:
        return False


def get_types(rows):
    types = []
    row1 = rows[0]
    types = []
    for val in row1:
        if is_num(val):
            types.append('real')
        else:
            types.append('text')
    return types


def get_refined_rows(rows, types):
    real_idx = []
    for i, type in enumerate(types):
        if type == 'real':
            real_idx.append(i)

    if len(real_idx) == 0:
        rrs = rows
    else:
        rrs = []
        for row in rows:
            rr = row
            for idx in real_idx:
                rr[idx] = float(row[idx])
            rrs.append(rr)
    return rrs


def csv_to_json(table_id, csv_file_name, json_file_name, working_folder='.'):
    csv_file_name = os.path.join(working_folder, csv_file_name)
    json_file_name = os.path.join(working_folder, json_file_name)
    with open(csv_file_name) as f:
        cf = csv.DictReader(f, delimiter=',')
        record = {}
        record['header'] = [(name or 'col{}'.format(i)) for i, name in enumerate(cf.fieldnames)]
        record['page_title'] = None
        record['id'] = table_id
        record['caption'] = None
        record['rows'] = [list(row.values()) for row in cf]
        record['name'] = get_table_name(table_id)

        # infer type based on first row

        record['types'] = get_types(rows=record['rows'])
        refined_rows = get_refined_rows(rows=record['rows'], types=record['types'])
        record['rows'] = refined_rows

    # save
    with open(json_file_name, 'a+') as fout:
        json.dump(record, fout)
        fout.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    #parser.add_argument('file', metavar='file.csv')
    parser.add_argument('file', metavar='all_utf8.csv')
    #working_folder = './data_and_model'
    working_folder = './gagaQA'
    args = parser.parse_args()
    table_id = os.path.splitext(os.path.basename(args.file))[0]
    csv_to_sqlite(table_id, args.file, '{}.db'.format(args.split), working_folder)
    csv_to_json(table_id, args.file, '{}.tables.jsonl'.format(args.split), working_folder)
    print("Added table with id '{id}' (name '{name}') to {split}.db and {split}.tables.jsonl".format(
        id=table_id, name=get_table_name(table_id), split=args.split))
