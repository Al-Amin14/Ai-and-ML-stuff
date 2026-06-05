
import requests


def get_embeddings(text_list):
    res = requests.post("http://localhost:11434/api/embed",json={
        "model":"bge-m3",
        "input":text_list
    })
    

    embeddings=res.json()['embeddings']

    return embeddings