# Système RAG Modulaire avec LLM Local

**Cette RAG est mise en place et maintenue par les élèves du Centrale Digital Lab 2025.**

Ce projet est une implémentation complète et modulaire d'un système de **Retrieval-Augmented Generation (RAG)**. Il est conçu pour fonctionner localement, en utilisant des modèles open-source (LLM et embedding) et en offrant une suite d'outils en ligne de commande pour gérer l'ensemble du cycle de vie des données, de l'ingestion à l'interrogation.

##  Fonctionnalités

-   **Ingestion de Données Polyvalente :** Prise en charge de multiples formats de fichiers (`.pdf`, `.docx`, `.doc`, `.txt`, `.md`) grâce à un `DocumentProcessor` robuste. [[1]](file://document_processor.py)
-   **OCR Intelligent :** Détecte automatiquement les PDF scannés ou de mauvaise qualité et applique un traitement OCR avec Tesseract pour en extraire le texte. [[2]](file://ocr_processor.py)
-   **Cache de Traitement :** Un système de cache intelligent basé sur le hash des fichiers évite de retraiter les documents qui n'ont pas changé, accélérant considérablement les mises à jour. [[1]](file://document_processor.py)
-   **Interface de Gestion Complète :** Un script interactif (`manage_collections_langchain.py`) permet de créer, supprimer, renommer, et gérer les collections de la base de données vectorielle ChromaDB. [[3]](file://manage_collections_langchain.py)
-   **Support Multi-Modèles d'Embedding :** Changez dynamiquement de modèle d'embedding (MPNet, MiniLM, Qwen) pour chaque collection et comparez leurs performances. [[4]](file://vectorizor.py)
-   **Outil de Benchmark Intégré :** Un script de benchmark (`benchmark_collection_langchain.py`) permet de tester et comparer les performances (score de pertinence, vitesse) de différentes configurations de collections. [[5]](file://benchmark_collection_langchain.py)
-   **Module de Re-ranking :** Améliore la pertinence des résultats en combinant le score sémantique initial avec un score lexical heuristique.
-   **Installation et Lancement Automatisés :** Des scripts `start.sh` et `start.bat` vérifient et installent les dépendances système, téléchargent le LLM et lancent le serveur, rendant le projet facile à déployer.
-   **Interface d'Interrogation :** Un script de test (`test_rag2.py`) fournit un mode "chat" interactif pour poser des questions à vos bases de connaissances. [[6]](file://test_rag2.py)

##  Installation

Ce projet utilise **Conda** pour gérer l'environnement Python. Les scripts d'installation automatisent la mise en place des dépendances système (LibreOffice, Tesseract) et le téléchargement du modèle LLM.

### Pré-requis

-   [Git](https://git-scm.com/)
-   [Anaconda](https://www.anaconda.com/download)
-   **(Pour Windows)** Il est recommandé d'installer [Chocolatey](https://chocolatey.org/install). Le script vous guidera si besoin.

### Instructions

1.  **Clonez ce dépôt :**
    ```bash
    git clone https://votre-url-de-depot/RAG_CDL25.git
    cd RAG_CDL25
    ```

2.  **Lancez le script de démarrage correspondant à votre système :**

    -   #### Pour Linux (Ubuntu/Debian) ou macOS :
        Ouvrez un terminal et lancez :
        ```bash
        chmod +x start.sh
        ./start.sh
        ```
        *(Le script vous demandera votre mot de passe `sudo` pour installer les paquets système si nécessaire).*

    -   #### Pour Windows :
        Ouvrez une invite de commande (`cmd.exe`) **en tant qu'administrateur** et lancez :
        ```batch
        start.bat
        ```
        *(Le script utilisera Chocolatey pour installer LibreOffice et Tesseract si nécessaire).*

Le script va automatiquement :
- Vérifier et installer LibreOffice et Tesseract.
- Télécharger le modèle LLM dans le dossier `models/`.
- Créer l'environnement Conda `ragcdl` s'il n'existe pas.
- Lancer le serveur LLM.

##  Guide de Démarrage Rapide

L'utilisation du projet se fait en **deux terminaux parallèles**.

### Terminal 1 : Lancer le Serveur LLM

C'est la première chose à faire. Le serveur doit tourner en permanence pour que le RAG puisse lui envoyer des requêtes.

```bash
./start.sh
```
*(ou `start.bat` sur Windows)*

Laissez ce terminal ouvert. Il va afficher les logs du serveur LLM.

### Terminal 2 : Gérer les Données et Interroger le RAG

Ouvrez un **nouveau terminal**.

1.  **Activez l'environnement Conda :**
    ```bash
    conda activate ragcdl
    ```

2.  **Créez et gérez vos bases de connaissances :**
    Utilisez le script de gestion pour créer votre première collection de vecteurs.
    ```bash
    python manage_collections_langchain.py
    ```
    Suivez le menu interactif pour créer une collection à partir d'un dossier de documents (par exemple, `code/base_test/DATA_Test`).

3.  **Posez une question à votre RAG :**
    Le script `test_rag2.py` est le point d'entrée pour interroger vos données.
    ```bash
    python test_rag2.py
    ```
    Le script vous guidera à travers les étapes suivantes :
    -   Il listera toutes vos collections existantes.
    -   Vous pourrez choisir celle que vous souhaitez interroger.
    -   Il vous proposera ensuite un **mode interactif (chat)** pour poser vos questions et recevoir des réponses directement de votre RAG.

4.  **Comparez vos configurations :**
    Pour évaluer la performance de vos différentes collections, utilisez l'outil de benchmark.
    ```bash
    python benchmark_collection_langchain.py
    ```

##  Structure du Projet

Le projet est organisé de manière modulaire pour faciliter la maintenance et l'évolution.

```
.
├── code/
│   ├── base_test/         # Données de test
│   └── test_texts/        # Sorties de test OCR
├── models/                # Modèles LLM téléchargés (ignoré par Git)
├── processed_texts/       # Cache des textes extraits (ignoré par Git)
├── chroma_db_local/       # Base de données vectorielle locale (ignoré par Git)
├── config.py              # Fichier de configuration central
├── document_processor.py  # Logique d'extraction et de cache des documents
├── ocr_processor.py       # Logique d'OCR pour les PDF
├── retrival_langchain.py  # Cœur de la logique RAG (retrieval, chunking)
├── rag.py                 # Classe principale assemblant les composants
├── manage_collections_langchain.py # Interface de gestion des collections
├── benchmark_collection_langchain.py # Outil de benchmark
├── test_rag2.py           # Script pour interroger le RAG
├── environment.yml        # Dépendances de l'environnement Conda
├── start.sh               # Script de lancement pour Linux/macOS
├── start.bat              # Script de lancement pour Windows
└── README.md              # Ce fichier
```

##  Configuration

Les principaux paramètres du projet peuvent être modifiés dans le fichier `config.py` [[7]](file://config.py):

-   `DEFAULT_DATA_DIR`: Le dossier par défaut contenant les documents sources.
-   `DEFAULT_EMBEDDING_MODEL`: Le modèle d'embedding Hugging Face utilisé par défaut.
-   `DEFAULT_LLM_MODEL`: Le nom du fichier du modèle LLM à utiliser.
-   `LLM_BASE_URL`: L'URL et le port où le serveur LLM est accessible.

## Licence

Ce projet est sous licence MIT.

