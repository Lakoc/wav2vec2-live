# automatic speech recognition with wav2vec2 

Use any wav2vec model with a microphone.

![demo gif](./docs/wav2veclive.gif)

## Setup

I recommend to install this project in a virtual environment.

```
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

Depending on linux distribution you might encounter an **error that portaudio was not found** when installing pyaudio. For Ubuntu you can solve that issue by installing the "portaudio19-dev" package.

```
sudo apt install portaudio19-dev
```

Download model from drive (https://drive.google.com/drive/folders/1hveY9Y3U0eBAP-qTKGzIcqzJZSilI-s-?usp=share_link) and unzip it.


Start demo app with relative path to unzipped model.

```
python demo_app.py --model checkpoint-115000
```