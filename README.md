# Proyecto de Tesis: Fine-tuning del Modelo Whisper para Transcripción en Español

Este proyecto se centra en el entrenamiento y ajuste fino del modelo Whisper de OpenAI para la tarea de transcripción de audio en español. Todo el proyecto está desarrollado en Python.

## Descripción

El objetivo de este proyecto es ajustar el modelo Whisper para transcribir audio en español. Utiliza datasets de entrenamiento y prueba que contienen pares de audio y texto.

## Requisitos

- Python 3.8 o superior
- pip (Gestor de paquetes de Python)

## Instalación

1. **Clonar el repositorio**
    ```sh
    git clone "Repo"
    cd "repo"
    ```

2. **Descargar y preparar los datasets**

    Coloca los archivos `train.csv` y `test.csv` en el directorio del proyecto.

## Dependencias

Asegúrate de tener instaladas las siguientes dependencias en tu entorno de Python:

- `transformers`
- `datasets`
- `evaluate`
- `pandas`
- `torch`
- `soundfile`
- `collections`
- `pydub`
- `numpy`

Estas dependencias se pueden instalar mediante el archivo `requirements.txt` mencionado en la sección de instalación.

## Uso

### Entrenamiento del Modelo

1. **Cargar los datasets**
    ```python
    from datasets import Dataset, concatenate_datasets, load_from_disk
    import pandas as pd
    from datasets import Audio
    import gc
    import os
    from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
        Seq2SeqTrainingArguments, Seq2SeqTrainer
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    import torch
    import evaluate

    # Cargar los datos de entrenamiento y prueba desde archivos CSV
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Renombrar las columnas para que sean consistentes
    train_df.columns = ["audio", "sentence"]
    test_df.columns = ["audio", "sentence"]

    # Convertir los DataFrames de pandas a objetos Dataset de Hugging Face
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Definir la frecuencia de muestreo para los datos de audio
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Cargar los componentes necesarios del modelo Whisper de Hugging Face
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="Spanish", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="Spanish", task="transcribe")

    # Función para preparar el dataset
    def prepare_dataset(examples):
        # Extraer características de audio
        audio = examples["audio"]
        examples["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]

        # Tokenizar las sentencias
        sentences = examples["sentence"]
        examples["labels"] = tokenizer(sentences).input_ids
        return examples

    # Dividir y procesar el dataset en fragmentos más pequeños y guardarlos en disco
    def process_in_batches(dataset, prepare_function, save_path):
        os.makedirs(save_path, exist_ok=True)
        temp_datasets = []
        for i in range(0, len(dataset), batch_size):
            subset = dataset.select(range(i, min(i + batch_size, len(dataset))))
            processed_subset = subset.map(prepare_function, num_proc=1)
            temp_path = os.path.join(save_path, f"batch_{i // batch_size}")
            processed_subset.save_to_disk(temp_path)
            temp_datasets.append(temp_path)
            del subset, processed_subset
            gc.collect()
        return temp_datasets

    # Definir el tamaño de los lotes y las rutas de guardado
    batch_size = 500
    train_save_path = "train_temp"
    test_save_path = "test_temp"

    # Procesar y guardar los datasets en fragmentos
    train_temp_datasets = process_in_batches(train_dataset, prepare_dataset, train_save_path)
    test_temp_datasets = process_in_batches(test_dataset, prepare_dataset, test_save_path)

    # Cargar los datasets procesados desde el disco y concatenarlos
    def load_temp_datasets(temp_paths):
        datasets = [load_from_disk(path) for path in temp_paths]
        return concatenate_datasets(datasets)

    train_dataset = load_temp_datasets(train_temp_datasets)
    test_dataset = load_temp_datasets(test_temp_datasets)

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    # Definir el data collator para el entrenamiento
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Cargar el métrico WER (Word Error Rate)
    metric = evaluate.load("wer")

    # Función para computar las métricas de evaluación
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Cargar el modelo preentrenado Whisper
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model.config.language = "spanish"
    model.config.task = "transcribe"
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Definir los argumentos de entrenamiento
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-finetuned-es",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=100,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        fp16=False,
        evaluation_strategy="steps",
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=25,
        eval_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    # Instanciar el entrenador
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # Entrenar el modelo
    trainer.train()
    ```

### Transcripción de Audio utilizando los datos nuevos

1. **Cargar las rutas de los checkpoints**
    ```python
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
    ```

## Dataset Utilizado

El dataset utilizado para este proyecto es el TEDx Spanish Corpus, que se puede descargar desde el siguiente [enlace](https://www.openslr.org/67/).

**Información del dataset:**

- **Identificador**: SLR67
- **Resumen**: Datos en español tomados de las charlas TEDx.
- **Categoría**: Speech
- **Licencia**: Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)
- **Descargas**: 
  - [tedx_spanish_corpus.tgz (2.3G)](https://www.openslr.org/resources/67/tedx_spanish_corpus.tgz) (Mirrors: [US](https://www.openslr.org/resources/67/tedx_spanish_corpus.tgz) [EU](https://www.openslr.org/resources/67/tedx_spanish_corpus.tgz) [CN](https://www.openslr.org/resources/67/tedx_spanish_corpus.tgz))

**Acerca de este recurso:**

El TEDx Spanish Corpus es un corpus de género desequilibrado de 24 horas de duración. Contiene discursos espontáneos de varios expositores en eventos TEDx; la mayoría son hombres. Las transcripciones se presentan en minúsculas sin signos de puntuación.

El proceso de recopilación de datos fue desarrollado en parte por el programa de servicio social "Desarrollo de Tecnologías del Habla" que depende de la Universidad Nacional Autónoma de México y en parte por el proyecto CIEMPIESS-UNAM.

Agradecimientos especiales al equipo de TED-Talks por permitirnos compartir este dataset.

**Cita del dataset:**

@misc{mena_2019,
	title = "{TEDx Spanish Corpus. Audio and transcripts in Spanish taken from the TEDx Talks; shared under the CC BY-NC-ND 4.0 license}",
	author = "Hernandez-Mena, Carlos D.",
	howpublished = "Web Download",
	institution = "Universidad Nacional Autonoma de Mexico",
	location = "Mexico City",
	year = "2019"
}

## Configuración del Sistema

Este programa se ejecutó inicialmente en la siguiente configuración de hardware y software:

- **Procesador**: AMD Ryzen 7 3700X 8-Core Processor 3.60 GHz
- **RAM**: 32.0 GB
- **Sistema Operativo**: 64-bit, x64-based processor
- **GPU**: NVIDIA GeForce RTX 3060
- **Almacenamiento**: 1.5 TB
- **Sistema operativo**: Windows

## Contribución

Si deseas contribuir a este proyecto, por favor sigue los siguientes pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -am 'Agrega nueva funcionalidad'`).
4. Sube los cambios a tu rama (`git push origin feature/nueva-funcionalidad`).
5. Abre un Pull Request.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Para más información, consulta el archivo `LICENSE`.
