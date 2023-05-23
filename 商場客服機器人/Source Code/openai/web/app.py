from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import io
import os
import sys
import wave
import tempfile
from pydub import AudioSegment
import sqlite3
import opencc
import json
from collections import OrderedDict
import openai
import pandas as pd
import numpy as np
import pickle
from transformers import GPT2TokenizerFast
from typing import List
import re

app = Flask(__name__)

openai.api_key = "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"
COMPLETIONS_MODEL = "babbage"

# 創建 OpenCC 的簡體中文轉換器
t2s_converter = opencc.OpenCC('t2s')
s2t_converter = opencc.OpenCC('s2t')

table_schema = {0: "main_category", 1: "sub_category", 2: "product_type", 3: "weight", 4: "item", 5: "price", 6: "unit_price", 7: "location"}
agg_sql_dict = {0: "", 1: "avg", 2: "max", 3: "min", 4: "count", 5: "sum"}
col_mapping_table = {"main_category":"主分類", "sub_category":"次分類", "product_type":"商品種類", "weight":"重量", "item":"品項", "price":"價格", "unit_price":"單價", "location":"位置"}

def map_query(sql_query, table_schema):
    # 提取数字和条件值
    numbers = re.findall(r'\b\d+\b', sql_query)
    condition_value = re.findall(r"'([^']*)'", sql_query)[0]
    
    # 替换数字和条件值
    for number in numbers:
        number = int(number)
        if number in table_schema:
            field = table_schema[number]
            sql_query = re.sub(r'\b{}\b'.format(number), field, sql_query)
    
    sql_query = re.sub(r"{}\s*=\s*'\b([^']+)\b'".format(table_schema[1]), "{} = '{}'".format(table_schema[1], condition_value), sql_query)
    
    return sql_query     

def get_sql_by_question_fm_openai(question):
    retrieve_response = openai.FineTune.retrieve("ft-tAQHNK9QLPy2zwr9yI3GytbS")
    fine_tuned_model = retrieve_response.fine_tuned_model
    prompt = question + " ->"
    answer = openai.Completion.create(
        model=fine_tuned_model,
        prompt=prompt,
        max_tokens=50,
        temperature=0.1
    )
    # 使用 "\n" 作為分割符號，將文字串分割成多個部分
    parts = answer['choices'][0]['text'].split("\n")

    # 取得第一個部分
    first_part = parts[0]

    print(first_part)

    sql_query = "select" + first_part

    return map_query(sql_query, table_schema)

def t2s(question):
    simplified_chinese_text = t2s_converter.convert(question)
    return simplified_chinese_text   

def s2t(sql):
    traditional_chinese_text = s2t_converter.convert(sql)
    return traditional_chinese_text

def get_cols_fm_sql_qry(sql_qry):
    # 使用正規表達式搜尋select和from之間的字串
    sel_cols = re.search('select (.*) from', sql_qry)
    
    # 如果有找到搜尋結果，將結果用逗號分割
    if sel_cols:
        selected_cols = sel_cols.group(1).split(', ')  

    for key, value in agg_sql_dict.items():
        for i in range(0, len(selected_cols)):
            selected_cols[i] = selected_cols[i].replace(value, "").replace(")", "").replace("(", "")
    selected_cols = [col_mapping_table[col.strip()] for col in selected_cols]
    return selected_cols

def beauti_sql_qry(sql_query):
    print(sql_query)

    if "min(" in sql_query:
        pos = sql_query.find("min(")
        sql_query = sql_query[:pos] + "product_type, location, price, " + sql_query[pos:]

    pos = sql_query.find('from product')
    # 檢查是否需要在 select 後面加上 product_type
    if pos >= 0:
        sql_query = sql_query.replace("from product", "from products", 1)

    pos = sql_query.find('select item from')
    # 檢查是否需要在 select 後面加上 product_type
    if pos >= 0:
        sql_query = sql_query.replace("select item", "select product_type, location, price ", 1)
        sql_query += " order by price"

    pos = sql_query.find('select price from')
    # 檢查是否需要在 select 後面加上 product_type
    if pos >= 0:
        sql_query = sql_query.replace("select price", "select product_type, location, price ", 1)
        sql_query += " order by price"

    pos = sql_query.find('select location from')
    # 檢查是否需要在 select 後面加上 product_type
    if pos >= 0:
        sql_query = sql_query.replace("select location", "select product_type, location, price ", 1)
        sql_query += " order by price"

    pos = sql_query.find('order')
    if pos < 0:
        sql_query += " order by 1"
    return sql_query

@app.route('/get-db-data', methods=['POST'])
def get_db_data():
    data = request.get_json()
    simplified_chinese_text = t2s(data['question'])
    
    # 將 Python 對象轉換為 JSON 字串
    data = {"question": simplified_chinese_text, "table_id": "6c11e43d2fb211ebbb1a7315e321f5c5"}

    sql_qry = beauti_sql_qry(s2t(get_sql_by_question_fm_openai(t2s(data['question']))))
    cols = get_cols_fm_sql_qry(sql_qry)
    
    conn = sqlite3.connect('customer.db')
    cur = conn.cursor()
    cur.execute(sql_qry)
    rows = cur.fetchall()

    rows = list(OrderedDict.fromkeys(rows))
    conn.close()
    return jsonify({'data': rows, 'cols':cols, 'sql':sql_qry})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    audio_file = request.files.get('audio')
    # 取得 app.py 的資料夾路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 將 WebM 轉換為 WAV
    with tempfile.NamedTemporaryFile(suffix='.wav', dir=current_dir, delete=False) as temp_wav_file:
        with wave.open(temp_wav_file.name, 'wb') as wav_file:
            audio = AudioSegment.from_file(io.BytesIO(audio_file.read()), format='webm')
            wav_file.setnchannels(audio.channels)
            wav_file.setsampwidth(audio.sample_width)
            wav_file.setframerate(audio.frame_rate)
            wav_file.writeframesraw(audio._data)
        
        # 使用 SpeechRecognition 模組進行語音轉換
        recognizer = sr.Recognizer()
        # with sr.AudioFile("test_voice_babycorn.wav") as source:
        with sr.AudioFile(temp_wav_file.name) as source:
            audio_data = recognizer.record(source)
            # transcription = recognizer.recognize_google(audio_data, language='cmn-Hans-CN')
            transcription = recognizer.recognize_google(audio_data, language='zh-TW')
            return jsonify({'transcription': transcription})

if __name__ == '__main__':
    app.run(debug=True)