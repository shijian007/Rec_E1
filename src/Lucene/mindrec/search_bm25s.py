import os, json
import bm25s
import Stemmer
from typing import List, Dict

corpus_data_dir = '/opt/data1/xiezhijun/code/Rec-E1/database/rl_rec/mind_small_v1d1_153k_0423/jsonl_docs/'
method = ''
if method == 'bm25+':
    bm25s_model_path = '/opt/data1/xiezhijun/code/Rec-E1/database/rl_rec/mind_small_v1d1_153k_0423/bm25s/mindsmall_bm25jia_model_0514'
else:
    bm25s_model_path = '/opt/data1/xiezhijun/code/Rec-E1/database/rl_rec/mind_small_v1d1_153k_0423/bm25s/mindsmall_bm25s_model_0514'


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def save_jsonl(data, sfile):
    with open(sfile, 'w', encoding='utf-8') as file:
        for entry in data:
            json_str = json.dumps(entry)
            file.write(json_str + '\n')

def list_files(directory):
    all_files = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path):
            all_files.append(full_path)
    return all_files

def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


class BM25SModel:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        if method:
            self.retriever = bm25s.BM25(method="bm25+", delta=1.5)
        else:
            self.retriever = bm25s.BM25(k1=k1, b=b)
        self.stemmer = Stemmer.Stemmer("english")
        self.item_map = {}
        self.init_model()

    def init_model(self):
        try:
            data = []
            data_files = list_files(corpus_data_dir)
            for rfile in data_files:
                data.extend(read_jsonl(rfile))
            self.init_item_map(data)
        except:
            return 'Load corpus data Error!'

        try:
            self.load_model(bm25s_model_path)
        except:
            return 'Load bm25s model Error!'

    def init_item_map(self, items: List[Dict]):
        self.item_map = {idx: item['id'] for idx, item in enumerate(items)}

    def add_documents(self, items: List[Dict]):
        self.init_item_map(items)
        corpus = [item['contents'] for item in items]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=self.stemmer, return_ids=True)

        self.retriever.index(corpus_tokens)

    def batch_search(self, queries: list, k: int = 5000, threads: int = 1) -> List[list]:
        return self.search(queries[0], k=k)

    def search(self, query: str, k: int = 5000) -> List[list]:
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        results, scores = self.retriever.retrieve(query_tokens, k=k)

        return [[self.item_map[results[0, i]]] for i in range(results.shape[1])]

    def save_model(self, save_path):
        mkdir(save_path)
        self.retriever.save(save_path)

    def load_model(self, load_path):
        self.retriever = bm25s.BM25.load(load_path)


if __name__ == "__main__":
    # load data
    # data = [
    #     {'itemid': 'N55528', 'content': 'Shop the notebooks, jackets, and more that the royals can\'t live without.'},
    #     {'itemid': 'N55529', 'content': 'Royal family approved sleep items for autumn and spring.'},
    #     {'itemid': 'N55520', 'content': 'Bert family approved fashion items for spring season and summer.'},
    #     {'itemid': 'N55521', 'content': 'snake approved fashion items for winter season and sleep.'},
    #     {'itemid': 'N55530', 'content': 'Exclusive interview with the royal designer about jacket trends.'}
    # ]

    data = []
    data_files = list_files(corpus_data_dir)
    for rfile in data_files:
        data.extend(read_jsonl(rfile))
    print('data len:', len(data), data[:3])

    # setup index and save model
    bm25s_model = BM25SModel()
    bm25s_model.add_documents(data)
    bm25s_model.save_model(bm25s_model_path)
    print('model save done')

    # search query
    query = "cute animal AND service dog OR cat OR human interaction OR adoption crisis"
    bm25s_model2 = BM25SModel()
    results = bm25s_model2.batch_search([query], k=100)
    print('results:\n', results)
