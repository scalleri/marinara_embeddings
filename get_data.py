#!/usr/bin/env python3
"""
This module handles the input from a Elasticsearch source into VRT for the cwb
"""

from elasticsearch import Elasticsearch
import elasticsearch.helpers


# Setup main ES connection
es = Elasticsearch()


def get_documents(index, field, term,data_type,lang):
    """Gets a result from Elasticsearch via wildcard search

    Args:
        index: The target ES index to query
        field: the field to query (for instance: "source")
        term: the actual query term (for instance: "blick")

    Yields:
        dict: single result
    """
    query = {
        "query": {
            "wildcard": {
                field: term
            }
        }
    }

    res = elasticsearch.helpers.scan(
        es, index=index, query=query)  # , size=10000)

    print("Fetching results from Elasticsearch into RAM...")
    results = []
    for hit in res:
        try:
            
            data = hit['_source']['content'][data_type]
            if len(data) > 150 and len(data) <= 10000 and hit['_source']['text_lang']==lang:
                results.append(data)
        except KeyError:
            continue
    return results

def get_pos_doc(index, field, term,data_type,lang):
    """Gets a result from Elasticsearch via wildcard search

    Args:
        index: The target ES index to query
        field: the field to query (for instance: "source")
        term: the actual query term (for instance: "blick")

    Yields:
        dict: single result
    """
    query = {
        "query": {
            "wildcard": {
                field: term
            }
        }
    }

    res = elasticsearch.helpers.scan(
        es, index=index, query=query)  # , size=10000)

    print("Fetching results from Elasticsearch into RAM...")
    results = []
    for hit in res:
        try:
            
            data = hit['_source']['content'][data_type]
            pos = hit['_source']['content']['pos']
            if len(data) > 150 and len(data) <= 10000 and hit['_source']['text_lang']==lang:
                lemma_pos = [data[i]+"_"+pos[i] for i in range(0,len(data)-1) ]
                results.append(lemma_pos)
        except KeyError:
            continue
    return results
   