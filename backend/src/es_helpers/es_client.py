from elasticsearch import Elasticsearch
from datetime import datetime
import logging

class ElasticsearchClient:
    def __init__(self):
        self.es = Elasticsearch(['http://elasticsearch:9200'])
        self.index = 'api_security_logs'
        self.setup_index()

    def setup_index(self):
        mapping = {
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "endpoint": {"type": "keyword"},
                    "method": {"type": "keyword"},
                    "request": {
                        "properties": {
                            "headers": {"type": "object"},
                            "params": {"type": "object"},
                            "body": {"type": "object"}
                        }
                    },
                    "response": {
                        "properties": {
                            "status_code": {"type": "integer"},
                            "headers": {"type": "object"},
                            "body": {"type": "object"}
                        }
                    },
                    "schema": {"type": "object"}
                }
            }
        }

        try:
            if not self.es.indices.exists(index=self.index):
                self.es.indices.create(index=self.index, body=mapping)
        except Exception as e:
            logging.error(f"Error creating index: {str(e)}")

    def store_api_log(self, endpoint, method, request_data, response_data, schema=None):
        document = {
            "timestamp": datetime.now(),
            "endpoint": endpoint,
            "method": method,
            "request": request_data,
            "response": response_data,
            "schema": schema
        }

        try:
            self.es.index(index=self.index, document=document)
        except Exception as e:
            logging.error(f"Error storing log: {str(e)}")

    def get_api_history(self, endpoint=None):
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"endpoint": endpoint}} if endpoint else {"match_all": {}}
                    ]
                }
            },
            "sort": [{"timestamp": "desc"}],
            "size": 10
        }

        try:
            result = self.es.search(index=self.index, body=query)
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except Exception as e:
            logging.error(f"Error fetching logs: {str(e)}")
            return []