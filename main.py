import whisper
import torch
import sys
import os
import signal
from pyannote.audio import Pipeline

class TimeoutException(Exception):
    pass

def timeout_handler(_, __):
    raise TimeoutException
  
# Define o manipulador de sinal para o alarme
signal.signal(signal.SIGALRM, timeout_handler)

# Configurações de dispositivo (GPU ou CPU)
device = torch.device("vulkan" if torch.is_vulkan_available() else "cpu")
# Token de autenticação do Hugging Face
token = ""

# Verifica se o caminho do arquivo foi fornecido
if len(sys.argv) < 2:
    print("Uso: python script.py <caminho_do_audio> <numero_de_oradores=1>")
    sys.exit(1)

# Obtém o caminho do arquivo de áudio da linha de comando
audio_path = sys.argv[1]
num_of_speakers = int(sys.argv[2] if len(sys.argv) > 2 else 1)
print(f"Quantidade de oradores? {num_of_speakers}")
print(f"Processando áudio: {audio_path}")

# Verifica se o arquivo existe
if not os.path.isfile(audio_path):
    print("Erro: Arquivo não encontrado!")
    sys.exit(1)

# Lista para armazenar os segmentos transcritos
transcription = []

# Carrega o modelo Whisper
print("Carregando Whisper...")
model = whisper.load_model("medium").to(device)

if num_of_speakers < 2:
    # Processa todo o áudio de uma vez
    print("Transcrevendo todo o áudio...")
    result = model.transcribe(audio_path, language="pt")
    transcription.append(result["text"])
else:
    # Inicializa o modelo de diarização (precisa do token do Hugging Face)
    print("Carregando modelo de diarização...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token)

    # Roda a diarização no áudio
    print("Identificando falantes...")
    diarization = diarization_pipeline(audio_path, num_speakers=num_of_speakers)

    # Processa cada segmento de fala identificado
    print("Transcrevendo segmentos de fala...")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        try:
            # Define o tempo limite em segundos
            start, end = turn.start, turn.end
            print(f"Falante {speaker}: {start:.2f}s - {end:.2f}s")
            
            timeout = int((end - start) * 30)
            signal.alarm(timeout)

            # Extrai segmento de áudio
            result = model.transcribe(audio_path, language="pt", clip_timestamps=(start, end))

            # Adiciona à transcrição
            transcription.append(f"Falante {speaker}: {result['text']}")
            print(f"Falante {speaker}: {result['text']}")
            
             # Cancela o alarme se o processo terminar a tempo
            signal.alarm(0)
        except TimeoutException:
            print("O processo demorou muito e foi interrompido.")
        except Exception as e:
            print(f"Erro ao transcrever segmento: {e}")

# Salva o resultado em um arquivo de texto
output_path = os.path.splitext(audio_path)[0] + "_transcricao.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(transcription))

print(f"Transcrição completa salva em: {output_path}")