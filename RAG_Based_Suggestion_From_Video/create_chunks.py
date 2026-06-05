import whisper
import torch
import json
import os

model=whisper.load_model("small")
device="cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

audios=os.listdir('audio')
i=0
for audio in audios:
    number=audio.split('_')[0]
    title=audio.split('_')[1].split('.')[0]
    result=model.transcribe(audio= f"audio/{audio}",
                            language="hi",
                            task="translate",
                            word_timestamps=False
                            )
    chunks=[]
    
    for segment in result['segments']:
        chunks.append({"number": number, "title": title, "start": segment['start'], "end": segment['end'], "text": segment['text']})
    
    chunks_with_metadata={"text": result['text'], "chunks": chunks}
    
    with open(f"json/{audio}.json","w") as f:
        json.dump(chunks_with_metadata,f)
    i+=1
    print(f"Processed {i} : {audio}")