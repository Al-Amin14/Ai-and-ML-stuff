import requests
import json
import os
import pandas as pd
import joblib
from embedding_convertion import get_embeddings



json_files=os.listdir('json')

j=0
chunk_id=0
stor_dicts=[]
for json_file in json_files:
    with open(os.path.join('json', json_file), 'r') as f:
        data = json.load(f)
    # print([c['text'] for c in data['chunks']])
    # break
    text_list=[c['text'] for c in data['chunks']]
    embeddings=get_embeddings(text_list[:200])
    for i,chunk in enumerate(data['chunks'][:200]):
        chunk['chunk_id']=chunk_id
        chunk_id+=1
        chunk['embedding']=embeddings[i]
        stor_dicts.append(chunk)
    j+=1
    print(f"Processed {j} : {json_file}") 
    # break
    
    


df=pd.DataFrame.from_records(stor_dicts)

joblib.dump(df, "chunks_with_embeddings.joblib")


