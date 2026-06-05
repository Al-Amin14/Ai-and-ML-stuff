import whisper
import torch
import json

model=whisper.load_model("small")

device="cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

result=model.transcribe(audio= "audio/33_Exercise 4 - Multi Color Website.mp3",
                        language="hi",
                        task="translate",
                        word_timestamps=False
                        )

print(result['segments'])

chunks=[]

for segment in result['segments']:
    chunks.append({"start": segment['start'], "end": segment['end'], "text": segment['text']})
    print(segment)
    
print(chunks)

with open("output.json","w") as f:
    json.dump(chunks,f,indent=4)