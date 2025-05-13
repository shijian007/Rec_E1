from elasticsearch import Elasticsearch, helpers
import logging
import numpy as np
from datetime import datetime, timedelta

class Config:
    ES_HOSTS = ["http://127.0.0.1:9200"]
    INDEX = "mindsmall_test"


class ESClient:
    def __init__(self):
        self.es = Elasticsearch(hosts=Config.ES_HOSTS, timeout=30)
        if not self.es.ping():
            raise ConnectionError("Connect Elasticsearch Error!")
        self.index = Config.INDEX

    """set index"""
    def _setup_index(self):
        if not self.es.indices.exists(index=self.index):
            mapping = {
                "mappings": {
                    "properties": {
                        "itemid": {"type": "keyword"},
                        "tag": {"type": "keyword"},
                        "title": {"type": "text", "analyzer": "standard", "fields": {"keyword": {"type": "keyword"}}},
                        "abstract": {"type": "text", "analyzer": "standard", "fields": {"keyword": {"type": "keyword"}}}
                    }
                },
            }
            self.es.indices.create(index=self.index, body=mapping)

    """multi upsert"""
    def bulk_upsert(self, articles):
        actions = ({
            "_op_type": "update",
            "_index": self.index,
            "_id": article["itemid"],
            "doc": article,
            "doc_as_upsert": True
        } for article in articles)

        try:
            success, errors = helpers.bulk(self.es, actions, chunk_size=100,
                                           max_retries=3, initial_backoff=2, stats_only=False, raise_on_error=False)
            for error in errors:
                print("失败文档详情：", error)
        except Exception as e:
            print("异常信息：", e)

    """search"""
    def batch_search(self, queries, size=50, thread=1):
        query = {}
        query['query'] = queries[0]
        query['_source'] = ['itemid']
        query['size'] = size
        query['sort'] = [{"_score": {"order": "desc"}}]

        try:
            res = self.es.search(index=self.index, body=query)
            ret = [hit["_source"] for hit in res["hits"]["hits"]]
            return [[item['itemid']] for item in ret]
        except Exception as e:
            logging.error(f"搜索失败: {str(e)}")
            return []

# test
if __name__ == '__main__':
    # init
    print('-- init es --')
    client = ESClient()

    # search
    query = {
                "query": {
                    {
                        "multi_match": {
                            "query": "Trump Trump administration official book deal Russia investigation Biden Ukrainian president false claim investigation witness news",
                            "fields": ["title^1.5", "abstract^1.2", "tag"]
                        }
                    }
                }
            }
    res = client.batch_search([query])
    print(res)

