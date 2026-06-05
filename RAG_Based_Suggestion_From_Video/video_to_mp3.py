#Convert the vidoes to mp3
import os
import subprocess

files=os.listdir('video')
for file in files:
    tutorial_number=file.split('.')[0].split('#')[1]
    file_name=file.split(' _')[0]
    print(file)
    subprocess.run(['ffmpeg', '-i', f'video/{file}', f'audio/{tutorial_number}_{file_name}.mp3'])
