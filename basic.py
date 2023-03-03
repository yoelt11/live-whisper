import whisper


if __name__ == '__main__':
    
    model = whisper.load_model("base")
    
    # load audio and pad(trim it to fit 30 seconds
    audio = whisper.load_audio("dictate.wav")
    audio = whisper.pad_or_trim(audio)
    
    # make log-mel spectrogram 
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print result
    print(result.text)
