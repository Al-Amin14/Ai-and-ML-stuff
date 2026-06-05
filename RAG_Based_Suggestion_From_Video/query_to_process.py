
# import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from embedding_convertion import get_embeddings




def inference(prompt):
    r=requests.post('http://localhost:11434/api/generate',json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream":False
    })
    response=r.json()
    return response


df = joblib.load("chunks_with_embeddings.joblib")



incoming_query=input("Enter your query: ")
question_embedding=get_embeddings([incoming_query])[0]


similarities=cosine_similarity(np.vstack(df["embedding"]),[question_embedding]).flatten()
top_result=30
max_index=similarities.argsort()[::-1][:top_result]
new_df=df.iloc[max_index]


prompt=f'''I am teaching web development using Sigma web developmnet course. Here are video subtitle chunks containing video title, text, start time in second , end time in second , the text at that:


{(new_df[['title','number','text','start','end']]).to_json()}
-----------------------
{incoming_query}
user asked this question related to the video chunks, you have to answer in human way (do not mention the above formate) where and how  much content is taught where (in which video and at what timestamps) and guide the user to go to that particular video.
so generate piece of information of my question that i have pass to you
'''


    
result=inference(prompt)['response']

print(result)
with open("resultresponse.txt","w") as f:
    f.write(result)

