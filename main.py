import whisper
import torch
import sys
import os
from pyannote.audio import Pipeline
from pyannote.core import Segment

# Configurações de dispositivo (GPU ou CPU)
device = torch.device("vulkan" if torch.is_vulkan_available() else "cpu")
token = ""

# Verifica se o caminho do arquivo foi fornecido
if len(sys.argv) < 2:
    print("Uso: python script.py <caminho_do_audio>")
    sys.exit(1)

# Obtém o caminho do arquivo de áudio da linha de comando
audio_path = sys.argv[1]
print(f"Transcrevendo áudio: {audio_path}")

# Verifica se o arquivo existe
if not os.path.isfile(audio_path):
    print("Erro: Arquivo não encontrado!")
    sys.exit(1)

# Inicializa o modelo de diarização (precisa do token do Hugging Face)
print("Carregando modelo de diarização...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token)

# Roda a diarização no áudio
print("Identificando falantes...")
diarization = diarization_pipeline(audio_path)

# Carrega o modelo Whisper
print("Carregando Whisper...")
model = whisper.load_model("medium").to(device)

# Lista para armazenar os segmentos transcritos
transcription = []

# Processa cada segmento de fala identificado
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start, end = turn.start, turn.end
    print(f"Falante {speaker}: {start:.2f}s - {end:.2f}s")

    # Extrai segmento de áudio
    result = model.transcribe(audio_path, language="pt", clip_timestamps=(start, end))

    # Adiciona à transcrição
    transcription.append(f"Falante {speaker}: {result['text']}")

# Salva o resultado em um arquivo de texto
output_path = os.path.splitext(audio_path)[0] + "_transcricao.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(transcription))

print(f"Transcrição completa salva em: {output_path}")