import requests

url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

params = {
    "where": """location_postalcode LIKE '974%'
        AND firstdate_begin >= '2010-12-29T00:00:00'
        AND lastdate_end < '2026-12-30T00:00:00'
    """,
    "order_by": "firstdate_begin ASC",
    "limit": 100
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

print("Nombre de résultats :", data["total_count"])

for event in data["results"]:
    # filtre si none
    if event.get("title_fr") is not None:
        print('*'*50)
        print(
            event.get("title_fr"),
            event.get("firstdate_begin"),
            event.get("description_fr")
        )
        
# embedding
from embedding import embed_ollama, chunks_langchain

chunks_data = chunks_langchain(data['results'])
embed = embed_ollama(chunks_data)

print('-' * 50)
print('contenu apres embedding')
print(embed)