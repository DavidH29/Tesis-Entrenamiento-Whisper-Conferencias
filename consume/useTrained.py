from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from collections import Counter
import numpy as np
from pydub import AudioSegment

# Lista de rutas a los checkpoints
# ACLARACION: Estas rutas son personales según la PC en la que se entrenó el modelo, se recomienda
# modificar según su propia ruta

checkpoint_paths = [
    r"C:\Users\renec\PycharmProjects\speechrecognition\TrainignModel\Beta\whisper-finetuned-es\checkpoint-25",
    r"C:\Users\renec\PycharmProjects\speechrecognition\TrainignModel\Beta\whisper-finetuned-es\checkpoint-50",
    r"C:\Users\renec\PycharmProjects\speechrecognition\TrainignModel\Beta\whisper-finetuned-es\checkpoint-75",
    r"C:\Users\renec\PycharmProjects\speechrecognition\TrainignModel\Beta\whisper-finetuned-es\checkpoint-100"
]

# Cargar el procesador original
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# Función para cargar y preprocesar el audio
def load_and_preprocess_audio(file_path, target_sample_rate=16000):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(target_sample_rate)
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0  # Normalizar a rango [-1, 1]
    return samples

# Función para predecir utilizando un modelo finetuned
def transcribe_audio(model, file_path):
    # Cargar y procesar el audio
    audio_input = load_and_preprocess_audio(file_path)

    # Dividir el audio en fragmentos
    chunk_size = 16000 * 30  # 30 segundos de audio
    transcriptions = []
    confidence_scores = []

    for i in range(0, len(audio_input), chunk_size):
        audio_chunk = audio_input[i:i + chunk_size]

        # Procesar el fragmento de audio
        inputs = processor(audio_chunk, sampling_rate=16000, return_tensors="pt")

        # Generar la transcripción
        with torch.no_grad():
            predicted_ids = model.generate(inputs["input_features"], output_scores=True, return_dict_in_generate=True)
            logits = predicted_ids.scores[0]
            confidences = torch.nn.functional.softmax(logits, dim=-1)
            confidence_score = confidences.max().item()

        # Decodificar la transcripción
        transcription = processor.batch_decode(predicted_ids.sequences, skip_special_tokens=True)[0]
        transcriptions.append(transcription)
        confidence_scores.append(confidence_score)

    # Combinar transcripciones y promediar las puntuaciones de confianza
    full_transcription = " ".join(transcriptions)
    avg_confidence_score = np.mean(confidence_scores)

    return full_transcription, avg_confidence_score

# Función para verificar si una transcripción es repetitiva
def is_repetitive(text, threshold=0.2):
    words = text.split()
    most_common_word, count = Counter(words).most_common(1)[0]
    return count / len(words) > threshold

# Función para evaluar y seleccionar la mejor transcripción
def evaluate_transcriptions(transcriptions):
    valid_transcriptions = [(trans, score) for trans, score in transcriptions if not is_repetitive(trans)]

    if not valid_transcriptions:
        return "No valid transcription found."

    avg_length = np.mean([len(trans.split()) for trans, _ in valid_transcriptions])
    std_length = np.std([len(trans.split()) for trans, _ in valid_transcriptions])

    filtered_transcriptions = [
        (trans, score) for trans, score in valid_transcriptions
        if (avg_length - 2 * std_length) <= len(trans.split()) <= (avg_length + 2 * std_length)
    ]

    if not filtered_transcriptions:
        filtered_transcriptions = valid_transcriptions

    best_transcription = max(filtered_transcriptions, key=lambda x: x[1])[0]
    return best_transcription

# Ejemplo de uso con un nuevo archivo de audio
# ACLARACION: Estas rutas son personales según la PC en la que se entrenó el modelo, se recomienda
# modificar según su propia ruta
new_audio_path = r"C:\Users\renec\PycharmProjects\speechrecognition\TrainignModel\Beta\whisper-finetuned-es\CONIA AUDIOS\CONIA-2022-Sesión-11 SAMPLE 119.wav"

# Obtener transcripciones de cada checkpoint y almacenarlas con sus puntuaciones de confianza
transcriptions_with_confidences = []

for checkpoint_path in checkpoint_paths:
    # Cargar el modelo finetuned desde la carpeta del checkpoint
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
    print(f"Modelo cargado desde {checkpoint_path}")

    # Obtener la transcripción y la puntuación de confianza
    transcription, confidence_score = transcribe_audio(model, new_audio_path)

    print(f"Transcription from {checkpoint_path}: {transcription} with confidence score {confidence_score}")
    transcriptions_with_confidences.append((transcription, confidence_score))

# Evaluar y seleccionar la mejor transcripción
best_transcription = evaluate_transcriptions(transcriptions_with_confidences)

# Guardar el resultado final en un archivo txt
# ACLARACION: Definir ruta propia para almacenar el archivo TXT con la transcripcion final
output_file_path = r"C:\Users\renec\PycharmProjects\speechrecognition\TrainignModel\Beta\final_transcription.txt"

with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write("Final transcription result:\n")
    file.write(best_transcription)

# Mostrar el resultado final
print("Final transcription result:", best_transcription)
