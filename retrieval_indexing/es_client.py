from elasticsearch import Elasticsearch

# Password for the 'elastic' user generated by Elasticsearch
ELASTIC_PASSWORD = "ALcNNs5Wxa-d=Cgu6+_y"
PATH_TO_HTTP_CA_CRT = "/data1/xiaomanpan/tmp/http_ca.crt"

# Create the client instance
client = Elasticsearch(
    "https://localhost:9200",
    ca_certs=PATH_TO_HTTP_CA_CRT,
    basic_auth=("elastic", ELASTIC_PASSWORD)
)

# # Create the client instance
# client = Elasticsearch(
#     "https://10.12.192.31:9200",
#     ca_certs=PATH_TO_HTTP_CA_CRT,
#     basic_auth=("elastic", ELASTIC_PASSWORD)
# )


# Successful response!
print(client.info())