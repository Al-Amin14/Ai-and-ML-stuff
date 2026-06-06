# How to use this RAG ai for Asistant on you own
## Step 1: move all the mp4 file to folder

Move all you video file on video folder

## Step 2: video to mp 3 convertion

Run video_to_mp3.py convertion
## Step 3: mp3 to text convertion

Run mp3_to_text.py convertion (This basically help to convert mp3 to chunks in json folder , where each mp3 is allocated individual json file)

## Step 4: Convert chunks to vector 

Run chunks_to_vector.py , this will convert all to chunks of json file to embedding vector and store in joblib file

## Step 5: Query to final answer

This will take user query and store in give final answer by converting in embedding and find coscine similarity after that give that result to llama-3.2 model which in use local ollama , to give answer