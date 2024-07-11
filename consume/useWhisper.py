import whisper

# Se carga el modelo
model = whisper.load_model("base")

# Creamos el archivo para escribir los resultados
f = open("resultados.txt", "a", encoding="utf-8")

# Variables que se utilizan para cargar los audios
base = "conia_audios/CONIA-2022-Sesión-11 SAMPLE "
extension = ".wav"


# Esto itera desde 1 a 121, puesto que tenemos 121 audios
for x in range(99, 122):
    file = base + str(x) + extension
    # load audio and pad/trim it to fit 30 seconds
    print("Loading Audio: " + file)
    audio = whisper.load_audio(file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    #_, probs = model.detect_language(mel)
    #print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    print("Transcribing Audio")
    options = whisper.DecodingOptions(language="es")
    result = whisper.decode(model, mel, options)

    # print the recognized text
    f.write(result.text + "\n")
    print("Successfully Transcribed\n")
f.close()
print("All Audios transcribed")