from elasticsearch import Elasticsearch
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def verify_and_populate():
    es = Elasticsearch(['http://elasticsearch:9200'])

    # Check if index exists
    if es.indices.exists(index='api_security_logs'):
        # Get document count
        count = es.count(index='api_security_logs')
        logger.info(f"Current document count: {count['count']}")

        # Get sample document
        results = es.search(
            index='api_security_logs',
            body={
                "size": 1,
                "query": {"match_all": {}}
            }
        )
        if results['hits']['hits']:
            logger.info("Sample document:")
            logger.info(json.dumps(results['hits']['hits'][0]['_source'], indent=2))
        else:
            logger.warning("No documents found in index")
    else:
        logger.warning("Index does not exist")

if __name__ == "__main__":
    verify_and_populate()