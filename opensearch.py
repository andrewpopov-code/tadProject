from opensearchpy import OpenSearch
from opensearch_dsl import Search


host = 'localhost'
port = 9200
# Create the client with SSL/TLS and hostname verification disabled.
client = OpenSearch(
    hosts = [{'host': host, 'port': port}],
    http_compress = True, # enables gzip compression for request bodies
    use_ssl = False,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False
)

# define the query
query = {
    "query": {
        "multi_match": {
            "query": "Каковы сроки направления требования об уплате задолженности",
            'fields': ['Статья', 'Текст']
        }
    }
}
# search for documents in the 'movies' index with the given query
response = client.search(index='law-rag-index', body=query)
# extract the hits from the response
hits = response['hits']['hits']
# print the hits
for hit in hits:
    print(hit)


from scipy.ndimage import gaussian_filter
from scipy.stats import bootstrap
