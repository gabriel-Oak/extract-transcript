import whisper
import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Verifica se o caminho do arquivo foi fornecido
if len(sys.argv) < 2:
    print("Uso: python script.py <caminho_do_audio>")
    sys.exit(1)

# Obtém o caminho do arquivo de áudio da linha de comando
audio_path = sys.argv[1]
print(f"Transcrevendo áudio: {audio_path}")


# Carrega o modelo Whisper
print("Iniciando Whisper...")
model = whisper.load_model("medium").to(device)


# Transcreve o áudio
print("Transcrevendo...")
result = model.transcribe(audio_path, language="pt")

# Exibe o texto transcrito
print(result["text"])
