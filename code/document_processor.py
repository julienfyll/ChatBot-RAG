# document_processor.py

import os
import re
import json
import hashlib
import tempfile
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

from docx import Document
from ocr_processor import PDFOCRProcessor
from config import PROCESSED_TEXTS_DIR, ROOT_DATA_PATH

class DocumentProcessor:
    """
    Classe responsable de l'extraction de texte à partir de divers formats de fichiers.
    Elle intègre une logique de cache intelligente pour éviter de retraiter les fichiers
    qui n'ont pas changé.
    """
    def __init__(self, 
                 path_doc: Path = ROOT_DATA_PATH, 
                 processed_texts_dir: Path = PROCESSED_TEXTS_DIR):
        """
        Initialise le processeur de documents.

        Args:
            path_doc (Path): Le répertoire de base contenant les dossiers de données sources.
            processed_texts_dir (Path): Le répertoire où le cache des textes extraits sera stocké.
        """
        self.path_doc = Path(path_doc)
        self.processed_texts_dir = Path(processed_texts_dir)
        self.processed_texts_dir.mkdir(parents=True, exist_ok=True)
        
        self.ocr_processor = PDFOCRProcessor(lang='fra', dpi=300)
        
        # Gestion du cache
        self.cache_metadata = {}
        self.current_database_folder = None

        self._handlers = {
            '.txt': self._process_text,
            '.md': self._process_markdown,
            '.markdown': self._process_markdown,
            '.docx': self._process_docx,
            '.doc': self._process_doc,
            '.pdf': self._process_pdf,
        }

    # --- Méthodes de gestion du cache (privées) ---

    def _load_cache_metadata(self, database_folder: Path) -> dict:
        metadata_file = self.processed_texts_dir / database_folder / "database_infos" / ".metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_cache_metadata(self, database_folder: Path):
        metadata_dir = self.processed_texts_dir / database_folder / "database_infos"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = metadata_dir / ".metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache_metadata, f, indent=2, ensure_ascii=False)

    def _compute_file_hash(self, file_path: Path) -> str:
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()
    
    def _get_cached_text_path(self, source_key: str) -> Path:
        # La clé est déjà relative, on l'utilise pour construire le chemin du cache
        # ex: "DATA_Test/fichier.pdf" -> "processed_texts/DATA_Test/files/fichier.pdf.txt"
        path_parts = Path(source_key).parts
        database_folder = path_parts[0] if path_parts else "default"
        filename = Path(source_key).name
        cached_path = self.processed_texts_dir / database_folder / "files" / f"{filename}.txt"
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        return cached_path
    
    def _is_cache_valid(self, source_key: str, source_file_path: Path, cached_file: Path) -> bool:
        if not cached_file.exists(): 
            return False
        if source_key not in self.cache_metadata: 
            return False
    
        current_hash = self._compute_file_hash(source_file_path)        # On calcule le hash sur le fichier physique (chemin absolu)
        cached_hash = self.cache_metadata[source_key].get('hash')        # On compare avec le hash stocké sous la clé relative

        return cached_hash == current_hash

    def _save_text_to_cache(self, source_key: str, source_file_path: Path, text: str, method: str):
        database_folder = Path(source_key).parts[0]
        self._switch_cache_database(database_folder) # Utilise _switch_cache_database
        
        cached_path = self._get_cached_text_path(source_key)
        with open(cached_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # La clé du cache est le chemin relatif
        self.cache_metadata[source_key] = {
            'hash': self._compute_file_hash(source_file_path),
            'cached_at': datetime.now().isoformat(),
            'method': method,
            'text_length': len(text),
            'cached_file': str(cached_path.as_posix()),
            'database': str(database_folder)
        }
        self._save_cache_metadata(database_folder)
        print(f"  Cache sauvegardé : {database_folder}/{cached_path.name} ({method})")

    
    def _load_text_from_cache(self, source_key: str, source_file_path: Path) -> Optional[str]:
        database_folder = Path(source_key).parts[0]
        self._switch_cache_database(database_folder) # Utilise _switch_cache_database
        
        cached_path = self._get_cached_text_path(source_key)
        if not self._is_cache_valid(source_key, source_file_path, cached_path):
            return None
        
        with open(cached_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        method = self.cache_metadata.get(source_key, {}).get('method', 'unknown')
        print(f"  ✓ Cache utilisé : {database_folder}/{cached_path.name} ({method})")
        return text
    
    def _switch_cache_database(self, database_folder: str):
        """Change le contexte du cache vers une nouvelle base de données si nécessaire."""
        if self.current_database_folder != Path(database_folder):
            self.cache_metadata = self._load_cache_metadata(database_folder)
            self.current_database_folder = Path(database_folder)

    def _get_database_folder(self, source_file: Path) -> Path:
        try:
            relative_path = source_file.relative_to(self.path_doc)
            database_folder = relative_path.parts[0] if relative_path.parts else "default"
            return Path(database_folder)
        except ValueError:
            return Path("default")

    def _clean_markdown(self, text: str) -> str:
        text = re.sub(r'^\s*#+\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\!?\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'(\*\*|__)(.*?)(\1)', r'\2', text)
        text = re.sub(r'(\*|_)(.*?)(\1)', r'\2', text)
        text = re.sub(r'^\s*[\-\*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        return text

    # --- Méthode principale (publique) ---

    def process_documents(
        self,
        source: Optional[str] = None,
        fichiers_specifiques: Optional[List[str]] = None,
        force_reprocess: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Extrait le texte d'une liste de fichiers ou d'un dossier source.
        Utilise une logique de cache pour accélérer le traitement.
        
        Args:
            source (str, optional): Chemin vers un fichier/dossier source.
            fichiers_specifiques (List[str], optional): Liste de chemins de fichiers à traiter en priorité.
            force_reprocess (bool): Si True, ignore le cache et retraite tous les fichiers.
            
        Returns:
            Tuple[List[str], List[str]]: Un tuple contenant la liste des textes extraits et la liste de leurs chemins.
        """
        textes, chemins = [], []
        cache_hits, cache_misses = 0, 0

        if fichiers_specifiques:
            fichiers_a_traiter = [Path(f) for f in fichiers_specifiques]
        elif source and Path(source).is_file():
            fichiers_a_traiter = [Path(source)]
        elif source:
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"Le chemin '{source}' n'existe pas.")
            fichiers_a_traiter = list(source_path.rglob("*.*"))
        else:
            return [], []

        print(f"\n[DocumentProcessor] Traitement de {len(fichiers_a_traiter)} fichier(s)...")
        if not force_reprocess:
            print("[DocumentProcessor] Utilisation du cache activée.")

        for fichier in fichiers_a_traiter:
            if not fichier.is_file(): continue
            fichier_abs = fichier.resolve()
            chemin_normalise = fichier.as_posix()

            # On calcule le chemin relatif par rapport à la racine des données
            try:
                chemin_relatif = fichier.resolve().relative_to(self.path_doc.resolve()).as_posix()
            except ValueError:
                print(f"  /!\\ Avertissement : Le fichier {fichier.resolve()} n'est pas dans le ROOT_DATA_PATH ({self.path_doc.resolve()}). Il sera ignoré.")
                continue # On ignore ce fichier car on ne peut pas créer de chemin relatif fiable

            chemin_a_stocker = chemin_relatif
            
            if not force_reprocess:
                cached_text = self._load_text_from_cache(chemin_a_stocker, fichier)
                if cached_text is not None:
                    textes.append(cached_text)
                    chemins.append(chemin_a_stocker)
                    cache_hits += 1
                    continue
            
            cache_misses += 1
            print(f"  → Traitement de : {fichier.name}")

            handler = self._handlers.get(fichier.suffix.lower()) # Utiliser .lower() pour la robustesse
            
            if handler:
                try:
                    texte_extrait, methode = handler(fichier)
                    if texte_extrait and texte_extrait.strip():
                        self._save_text_to_cache(chemin_a_stocker, fichier, texte_extrait, methode)
                        textes.append(texte_extrait)
                        chemins.append(chemin_a_stocker)
                    else:
                        print(f"    /!\\ Fichier ignoré (aucun texte extrait) : {fichier.name}")
                except Exception as e:
                    print(f"    /!\\ Erreur lors du traitement de {fichier.name} avec le handler '{fichier.suffix}' : {e}")
            else:
                print(f"    /!\\ Type de fichier non supporté : {fichier.suffix}")
        
        print(f"[DocumentProcessor] Statistiques du cache : {cache_hits} hits, {cache_misses} misses.")
        return textes, chemins
    

    def _process_text(self, file_path: Path) -> Tuple[str, str]:
        """Stratégie de traitement pour les fichiers .txt."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text, "TXT_direct"

    def _process_markdown(self, file_path: Path) -> Tuple[str, str]:
        """Stratégie de traitement pour les fichiers .md."""
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        cleaned_text = self._clean_markdown(raw_text)
        return cleaned_text, "Markdown_cleaned"

    def _process_docx(self, file_path: Path) -> Tuple[str, str]:
        """Stratégie de traitement pour les fichiers .docx."""
        doc = Document(str(file_path))
        text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
        return text, "DOCX_direct"

    def _process_doc(self, file_path: Path) -> Tuple[str, str]:
        """Stratégie de traitement pour les fichiers .doc (via LibreOffice)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Utiliser check=True pour lever une exception en cas d'échec de LibreOffice
            subprocess.run(
                ['libreoffice', '--headless', '--convert-to', 'docx', '--outdir', temp_dir, str(file_path)],
                capture_output=True, text=True, timeout=60, check=True
            )
            docx_path = Path(temp_dir) / f"{file_path.stem}.docx"
            if docx_path.exists():
                # Réutilise la logique de .docx pour éviter la duplication de code
                return self._process_docx(docx_path)
        return None, "inconnue"

    def _process_pdf(self, file_path: Path) -> Tuple[str, str]:
        """Stratégie de traitement pour les fichiers .pdf (avec détection OCR)."""
        if not self.ocr_processor.has_text_layer(str(file_path)) or self.ocr_processor.detect_pdf_quality(str(file_path)):
            print("    → PDF scanné ou de mauvaise qualité, utilisation de l'OCR.")
            text = self.ocr_processor.ocr_pdf(str(file_path))
            method = "OCR_Tesseract"
        else:
            print("    → PDF avec texte de bonne qualité, extraction directe.")
            text = self.ocr_processor.extract_text_and_post_process(str(file_path))
            method = "PyPDF2_postprocessed"
        return text, method

        
    def clear_cache(self, database: Optional[str] = None):
        """
        Supprime le cache (soit une base de données spécifique, soit tout).

        Args:
            database (str, optional): Nom de la base à supprimer (ex: "DATA_Test").
                                      Si None, supprime TOUT le cache.
        """
        if database is not None:
            db_cache_dir = self.processed_texts_dir / database
            if not db_cache_dir.exists():
                print(f"  Cache pour la base '{database}' introuvable.")
                return
            
            shutil.rmtree(db_cache_dir)
            print(f"  Cache pour la base '{database}' supprimé.")
            
            if self.current_database_folder == Path(database):
                self.cache_metadata = {}
                self.current_database_folder = None
        else:
            shutil.rmtree(self.processed_texts_dir)
            self.processed_texts_dir.mkdir(parents=True, exist_ok=True)
            self.cache_metadata = {}
            self.current_database_folder = None
            print("✓ Cache complet supprimé.")

    def list_cache(self):
        """Affiche le contenu du cache organisé par base de données."""
        print("\n[DocumentProcessor] Contenu du cache des textes extraits")
        print("="*100)
        
        databases = [d for d in self.processed_texts_dir.iterdir() if d.is_dir()]
        if not databases:
            print("  Cache vide.")
            return

        total_files, total_size = 0, 0
        for db_folder in sorted(databases):
            json_file = db_folder / "database_infos" / ".metadata.json"
            if not json_file.exists(): continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except:
                continue
            
            if not metadata: continue

            print(f"\n  BASE : {db_folder.name}")
            print("-" * 50)
            
            db_size = 0
            for source_path, meta in metadata.items():
                cached_file = Path(meta['cached_file'])
                if cached_file.exists():
                    size_kb = cached_file.stat().st_size / 1024
                    db_size += size_kb
                    print(f"    - {Path(source_path).name} ({meta['method']}, {size_kb:.1f} KB)")
            
            print(f"\n    Sous-total : {len(metadata)} fichier(s), {db_size/1024:.1f} MB")
            total_files += len(metadata)
            total_size += db_size
        
        print("\n" + "="*100)
        print(f"  TOTAL : {total_files} fichier(s) dans {len(databases)} base(s), {total_size/1024:.1f} MB")
        print("="*100)

    def rebuild_cache_for_file(self, source_file: str):
        """
        Force le retraitement d'un fichier spécifique et met à jour son cache.

        Args:
            source_file (str): Chemin du fichier source à retraiter.
        """
        source_path = Path(source_file)
        if not source_path.exists():
            print(f"  Fichier introuvable : {source_file}")
            return

        print(f"  Retraitement forcé de : {source_path.name}")
        
        # On appelle simplement la méthode principale avec force_reprocess=True
        # pour ce fichier unique. La logique de suppression/mise à jour du cache
        # est déjà gérée par process_documents.
        self.process_documents(
            fichiers_specifiques=[str(source_path)],
            force_reprocess=True
        )
        print(f"  Fichier retraité et cache mis à jour.")

    def migrate_cache_paths_to_relative(self):
        """
        [MAINTENANCE] Parcourt tous les caches .metadata.json et convertit
        leurs clés en chemins relatifs par rapport au ROOT_DATA_PATH de l'instance.
        """
        print("\n[DocumentProcessor] Lancement de la migration des chemins du cache...")
        
        # S'assurer que le chemin racine est bien un objet Path résolu
        root_path = self.path_doc.resolve()
        print(f"  Chemin racine de référence : {root_path}")

        # Lister toutes les bases de données dans le cache
        databases = [d for d in self.processed_texts_dir.iterdir() if d.is_dir()]
        if not databases:
            print("  Cache vide, aucune migration nécessaire.")
            return

        total_keys_migrated = 0
        for db_folder in databases:
            metadata_file = db_folder / "database_infos" / ".metadata.json"
            if not metadata_file.exists():
                continue

            print(f"\n  Traitement de la base de cache : {db_folder.name}")
            
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    old_metadata = json.load(f)
            except json.JSONDecodeError:
                print(f"    /!\\ Fichier .metadata.json corrompu, ignoré.")
                continue

            new_metadata = {}
            keys_migrated_in_db = 0
            
            for old_key, value in old_metadata.items():
                try:
                    # On reconstruit le chemin absolu à partir de l'ancienne clé relative au projet
                    current_absolute_path = (Path.cwd() / old_key).resolve()
                    # On calcule le nouveau chemin relatif par rapport à ROOT_DATA_PATH
                    new_key = current_absolute_path.relative_to(root_path).as_posix()
                    
                    new_metadata[new_key] = value
                    if old_key != new_key:
                        keys_migrated_in_db += 1
                        
                except ValueError:
                    # Le chemin n'est pas sous la racine, on le garde tel quel (cas d'un chemin déjà propre)
                    new_metadata[old_key] = value
                except Exception as e:
                    print(f"    /!\\ Erreur inattendue pour la clé '{old_key}': {e}")
                    new_metadata[old_key] = value # On conserve l'ancienne clé en cas d'erreur

            if keys_migrated_in_db > 0:
                print(f"    {keys_migrated_in_db} clé(s) à mettre à jour.")
                # Sauvegarder le nouveau fichier de métadonnées
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(new_metadata, f, indent=2, ensure_ascii=False)
                print("    ✓ Fichier .metadata.json mis à jour.")
                total_keys_migrated += keys_migrated_in_db
            else:
                print("    ✓ Aucune clé à migrer dans cette base.")

        if total_keys_migrated > 0:
            print(f"\n✓ Migration du cache terminée. {total_keys_migrated} clé(s) mises à jour au total.")
        else:
            print("\n✓ Migration du cache terminée. Tous les chemins du cache étaient déjà corrects.")
