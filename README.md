# live-whisper
A place to test whisper by open-ai

### Install dependencies 
```cmd
pip3 install -r requirements.txt
```

### Scrips
* basic.py -  contains script for processing a single audio file (audio mus be prerecorded).
```python 
  audio = whisper.load_audio("dictate.wav") # modify audio file inside script accordingly
```

* live_whisper.py - transcribes audio in german or english
