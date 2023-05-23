# AIAFinalProject
AIA 期末專題

# 資料
1. TableQA (https://github.com/ZhuiyiTechnology/TableQA)
```
@misc{sun2020tableqa,
    title={TableQA: a Large-Scale Chinese Text-to-SQL Dataset for Table-Aware SQL Generation},
    author={Ningyuan Sun and Xuefeng Yang and Yunfeng Liu},
    year={2020},
    eprint={2006.06434},
    archivePrefix={arXiv},
    primaryClass={cs.DB}
}
```
2. wikisql

# 相關專案
1. 目前以 wikiSQL training, 且testing最高分的專案
https://github.com/microsoft/Table-Pretraining

2. bert + wikisql
https://github.com/naver/sqlova

# 開發Flow
1. speech to text (done...)
2. train bert by wikisql (surveying)
3. fine tune bert by cspider and pxgo data base on 2.'s model(cspider and pxgo data ready)
4. implement a web interface(not necessary)
5. combine 1. & 3. into 4.(not necessary)
