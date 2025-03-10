
import requests

def download_dropbox_file(dropbox_link, save_path):
    # Ensure the link is a direct download link
    direct_link = dropbox_link.replace("?dl=0", "?dl=1")

    response = requests.get(direct_link, stream=True)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Download complete: {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

# Example usage
UR_FUNNY_RAW = "https://www.dropbox.com/s/lg7kjx0kul3ansq/urfunny2_videos.zip?dl=1"
save_location = "ur-funny.zip"

download_dropbox_file(UR_FUNNY_RAW, save_location)

import zipfile
import os

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to: {extract_to}")

# Example usage
zip_file_path = "downloaded_file.zip"  # Path to the ZIP file
extract_folder = "dataset"  # Folder to extract files

# Ensure the extraction folder exists
os.makedirs(extract_folder, exist_ok=True)

unzip_file(save_location, extract_folder)

from audio_extract import extract_audio

extract_audio(input_path="./downloaded_file/urfunny2_videos/10.mp4", output_path="./audio",output_format='wav')

#extract_audio(input_path="./video.mp4",
#              output_path="./audio.mp3",
#              start_time="00:30",
#              overwrite=True)

from transformers import Wav2Vec2Model,Wav2Vec2FeatureExtractor
import librosa
import torch

input_audio, sample_rate = librosa.load("audio.wav",  sr=16000)

# can change to whatever variant of wav2vec2, have to check later
model_name = "facebook/wav2vec2-base-960h"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

i= feature_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate)
with torch.no_grad():
  o= model(i.input_values)
print(o.keys())
print(o.last_hidden_state.shape)
print(o.extract_features.shape)