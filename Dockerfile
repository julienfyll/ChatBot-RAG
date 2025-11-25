FROM python:3.12-slim

# 1. Installation des dépendances SYSTÈME (OCR, LibreOffice, etc.)
# C'est lourd mais nécessaire pour ton ocr_processor.py et document_processor.py
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-fra \
    libreoffice \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Configuration du dossier de travail
WORKDIR /app

# 3. Copie des requirements et installation Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copie de tout le code source
COPY . .

# 5. Création des dossiers de données s'ils n'existent pas
RUN mkdir -p data/raw data/processed_texts chroma_db_local

# 6. Variable d'environnement pour dire à Python où trouver les modules
# Cela permet de faire "from src.rag import ..."
ENV PYTHONPATH=/app

# Le Flag pour settings.py
ENV RAG_ENVIRONMENT="docker"

# 7. Lancement de l'API
CMD ["python", "main.py"]