
import librosa
import os
# for aifc issue for python 3.13+, install standard-aifc and standard-sunau
# can change to whatever variant of wav2vec2, have to check later
input_dir = './dataset/urfunny2_audios'

durations=[]
total = 0
min = 10000000
min_id = ""
max_id = ""
max = 0
total_time = 0.0
for filename in os.listdir(input_dir):
    duration = librosa.get_duration(path=input_dir + '/'+filename)
    durations.append(duration)
    total += 1
    total_time += duration
    if duration > max:
        max = duration
        max_id = filename
    if duration < min:
        min = duration
        min_id = filename

print(total_time/total)
print(total,max_id,max,min_id,min)