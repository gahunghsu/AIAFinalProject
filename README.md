# AIAFinalProject
AIA 期末專題

# 資料
1. CSpider, 簡體中文標註的 Text to SQL 數據集
https://github.com/taolusi/chisp
2. wikisql, 目前很多專案都用此數據集
https://github.com/salesforce/WikiSQL
3. 安裝 huggingface 的 datasets, 就可以使用wikisql的資料
```
pip install datasets
```

# 相關專案
1. 目前以 wikiSQL training, 且testing最高分的專案
https://github.com/microsoft/Table-Pretraining

2. bert + wikisql
https://github.com/naver/sqlova

# 開發Flow
1. speech to text
2. train bert by wikisql
3. train bert by cspider and pxgo data base on 2.'s model
4. implement a web interface
5. combine 1. & 3. into 4.
