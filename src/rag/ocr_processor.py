# ocr_processor.py

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import os
import PyPDF2
import re
from typing import List, Optional
from pathlib import Path

class PDFOCRProcessor:
    """Processeur OCR optimisé pour les PDFs scannés avec post-traitement ultra-agressif"""
    
    def __init__(self, lang='fra', dpi=300, output_dir='code/test_texts'):
        self.lang = lang
        self.dpi = dpi
        self.output_dir = output_dir
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(
                "Tesseract n'est pas installé ou non trouvé. "
                "Installez-le avec: sudo apt install tesseract-ocr tesseract-ocr-fra"
            )

    
    def _create_output_dir(self):
        """Crée le dossier de sortie s'il n'existe pas"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Dossier créé : {self.output_dir}")

    def has_text_layer(self, pdf_path: str) -> bool:
        """
        Vérifie si le PDF contient du texte encodé (couche texte)
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier {pdf_path} n'existe pas")
        print(f"Vérification de la présence de texte dans le PDF...")
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pages_to_check = min(3, len(pdf_reader.pages))
                for page_num in range(pages_to_check):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        print(f"✓ Texte encodé détecté (page {page_num + 1})")
                        return True
                print(f"✗ Aucun texte encodé trouvé (PDF scanné sans OCR)")
                return False
        except Exception as e:
            print(f" Erreur lors de la vérification : {e}")
            return False

    def detect_pdf_quality(self, pdf_path: str, min_words_in_line: int = 10, min_short_words: int = 5) -> bool:
        """
        Détecte si le PDF contient des mots morcelés (mauvaise qualité)
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier {pdf_path} n'existe pas")
        print(f"Analyse de la qualité du PDF...")
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if not text.strip():
                        continue
                    lines = text.split('\n')
                    for line in lines:
                        line_stripped = line.strip()
                        if not line_stripped:
                            continue
                        words = line_stripped.split()
                        if len(words) < min_words_in_line:
                            continue
                        consecutive_short = 0
                        for word in words:
                            if len(word) <= 2 and word.isalpha():
                                consecutive_short += 1
                                if consecutive_short >= min_short_words:
                                    print(f"✗ Mauvaise qualité détectée (page {page_num + 1})")
                                    print(f"  Ligne problématique : {line_stripped[:80]}...")
                                    return True
                            else:
                                consecutive_short = 0
                print(f"✓ Texte de bonne qualité")
                return False
        except Exception as e:
            print(f" Erreur lors de la lecture du PDF : {e}")
            return True

    def extract_text_and_post_process(self, pdf_path: str) -> str:
        """
        Extrait le texte d'un PDF avec PyPDF2 et applique le post-traitement
        """

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier {pdf_path} n'existe pas")
     
        print(f"Extraction du texte depuis le PDF (PyPDF2)...")
        full_text = ""

        try:
            with open(pdf_path, 'rb') as file:

                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                print(f"Nombre de pages : {total_pages}")

                for page_num in range(total_pages):

                    print(f"Extraction page {page_num + 1}/{total_pages}...")
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()

                    if page_text.strip():
                        full_text += f"\n=== Page {page_num + 1} ===\n\n{page_text}\n"

        except Exception as e:
            print(f" Erreur lors de l'extraction : {e}")
            raise

        if not full_text.strip():
            print("  Aucun texte extractible trouvé dans le PDF")
            return None
        
        print(f"✓ Texte extrait : {len(full_text)} caractères")
        print("Application du post-traitement...")
        cleaned_text = self.post_process_text(full_text)
        
        print(f"  Caractères avant post-traitement : {len(full_text)}")
        print(f"  Caractères après post-traitement : {len(cleaned_text)}")
        print(f"  Réduction : {100 - (len(cleaned_text) / len(full_text) * 100):.1f}%")

        return cleaned_text 

    # --- STRUCTURE DE POST-TRAITEMENT ---

    def post_process_text(self, text: str) -> str:
        """
        Applique une série de nettoyages et de normalisations au texte brut.
        Orchestre l'appel de plusieurs méthodes de post-traitement privées.
        """
        text = self._correct_prepositions_and_quotes(text)
        text = self._clean_ocr_artifacts(text)
        text = self._normalize_acronyms(text) 
        text = self._fix_spaced_letters(text)
        text = self._separate_uppercase_words(text)
        text = self._normalize_spacing_and_punctuation(text)
        text = self._remove_spurious_lines(text)
        text = self._finalize_text(text)
        return text

    def _correct_prepositions_and_quotes(self, text: str) -> str:
        """PHASE 1 : Corrige les apostrophes et les prépositions collées."""
        text = re.sub(r"''", "'", text)
        text = re.sub(r"``", '"', text)
        text = re.sub(r'à([lL]\')', r'à \1', text)
        text = re.sub(r'à([lL][aeu]|[uU]ne?|[cC]et)', r'à \1', text)
        text = re.sub(r'de([lL]\')', r'de \1', text)
        text = re.sub(r'de([lL][aeu]|[uU]ne?)', r'de \1', text)
        return text

    def _clean_ocr_artifacts(self, text: str) -> str:
        """PHASE 2 : Nettoie les artefacts spécifiques à l'OCR."""
        text = re.sub(r'^[ÙùFfEeIiLlBb]\s+(?=Page|Mars|Juin|Par\.)', '', text, flags=re.MULTILINE)
        text = re.sub(r'Pere?\.\s*Te', 'Pers. 602', text)
        text = re.sub(r'Pere?\.\s*44h', 'Pers. 445', text)
        text = re.sub(r'Pers\.\s*44h', 'Pers. 445', text)
        text = re.sub(r'^\s*[à;:.@]+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+[à;@]\s+', ' ', text)
        return text

    def _normalize_acronyms(self, text: str) -> str:
        """PHASE 3 : Normalise les acronymes avec des points mal espacés."""
        text = re.sub(r'E\.\s*D\.\s*F\.', 'E.D.F.', text)
        text = re.sub(r'G\.\s*D\.\s*F\.', 'G.D.F.', text)
        text = re.sub(r'I\.\s*V\.\s*D\.', 'I.V.D.', text)
        text = re.sub(r'I\.\s*G\.\s*E\.\s*C\.\s*O\.', 'I.G.E.C.O.', text)
        return text

    def _fix_spaced_letters(self, text: str) -> str:
        """PHASE 4 : Corrige les mots avec des lettres espacées."""
        lines = text.split('\n')
        fixed_lines = []
        for line in lines:
            if not line.strip():
                fixed_lines.append(line)
                continue
            words = line.split()
            fixed_words = []
            i = 0
            while i < len(words):
                word = words[i]
                if len(word) == 1 and word.isalpha():
                    collected = [word]
                    j = i + 1
                    while j < len(words) and len(words[j]) <= 2 and any(c.isalpha() for c in words[j]):
                        collected.append(words[j])
                        j += 1
                    if len(collected) >= 3:
                        merged = ''.join(collected)
                        fixed_words.append(merged)
                        i = j
                        continue
                fixed_words.append(word)
                i += 1
            fixed_line = ' '.join(fixed_words)
            for _ in range(2):
                before = fixed_line
                fixed_line = re.sub(r'\b([a-zà-û])\s+([a-zà-û])\b', r'\1\2', fixed_line, flags=re.IGNORECASE)
                if fixed_line == before:
                    break
            fixed_lines.append(fixed_line)
        return '\n'.join(fixed_lines)

    def _separate_uppercase_words(self, text: str) -> str:
        """PHASE 5 : Sépare les mots collés en majuscules."""
        def separator_logic(match):
            text_block = match.group(0)
            if '.' in text_block or text_block.count('-') > 1:
                return text_block
            common_words = ['MISSIONS', 'DETACHEMENTS', 'GESTION', 'AGENT', 'PENDANT', 'SITUATION', 'ADMINISTRATIVE', 'RETOUR', 'MISE', 'DISPOSITION', 'MISSION', 'COURTE', 'DUREE', 'LONGUE', 'DEFINITION', 'CHAPITRE', 'SOMMAIRE', 'RETRAITE', 'AVANCEMENTS', 'PERSONNEL', 'CONSULTATION']
            for word in sorted(common_words, key=len, reverse=True):
                if word in text_block:
                    text_block = text_block.replace(word, f' {word} ')
            text_block = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', text_block)
            text_block = re.sub(r'\s+', ' ', text_block)
            return text_block.strip()
        return re.sub(r'\b[A-ZÀ-Ÿ][A-ZÀ-Ÿa-z]{11,}\b', separator_logic, text)

    def _normalize_spacing_and_punctuation(self, text: str) -> str:
        """PHASE 6 & 7 : Normalise les espaces et la ponctuation."""
        text = re.sub(r' {2,}', ' ', text)
        lines = text.split('\n')
        normalized_lines = [line.strip() for line in lines]
        text = '\n'.join(normalized_lines)
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])([A-Za-zÀ-ÿ])', r'\1 \2', text)
        text = re.sub(r':([A-Za-z])', r': \1', text)
        text = re.sub(r'\.!+', '.', text)
        return text

    def _remove_spurious_lines(self, text: str) -> str:
        """PHASE 8 : Supprime les lignes courtes et parasites."""
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                cleaned_lines.append('')
                continue
            if len(line_stripped) > 20:
                cleaned_lines.append(line)
                continue
            if re.match(r'^[A-Z][a-z]+\s+\d{2}$', line_stripped): continue
            if re.match(r'^Par\.\s*\d+$', line_stripped): continue
            if re.match(r'^(?:Page|p\.?)\s*\d+$', line_stripped, re.IGNORECASE): continue
            if re.match(r'^[\s\-_=+*#~\.,:;!?—–]{3,}$', line_stripped): continue
            if len(line_stripped) <= 2 and not line_stripped.isdigit(): continue
            if re.match(r'^[A-Za-z\s]{1,5}$', line_stripped): continue
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def _finalize_text(self, text: str) -> str:
        """PHASE 9 : Nettoyage final des sauts de ligne."""
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        return text.strip()

    
    def ocr_image(self, image: Image.Image, preprocess: bool = False) -> str:
        """Effectue l'OCR sur une image"""
        if preprocess:
            image = self.preprocess_image(image)
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(image, lang=self.lang, config=custom_config)
        return text

    def ocr_pdf(self, pdf_path: str, preprocess: bool = False, post_process: bool = True) -> str:
        """Effectue l'OCR sur un PDF complet"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier {pdf_path} n'existe pas")
        print(f"Conversion du PDF en images (DPI={self.dpi})...")
        images = convert_from_path(pdf_path, dpi=self.dpi, thread_count=os.cpu_count() or 4)
        full_text = ""
        total_pages = len(images)
        for i, image in enumerate(images, 1):
            print(f"Traitement de la page {i}/{total_pages}...")
            page_text = self.ocr_image(image, preprocess=preprocess)
            if post_process:
                # Note: La méthode post_process_text est maintenant appelée ici
                page_text = self.post_process_text(page_text)
            if page_text.strip():
                full_text += f"\n=== Page {i} ===\n\n{page_text}\n"
        if post_process:
            print("Post-traitement final...")
            full_text = self.post_process_text(full_text)
        return full_text

    def ocr_pdf_to_txt(self, pdf_path: str, preprocess: bool = False, post_process: bool = True, output_filename: Optional[str] = None) -> str:
        """Effectue l'OCR sur un PDF et sauvegarde en .txt avec gestion des doublons"""
        self._create_output_dir()
        text = self.ocr_pdf(pdf_path, preprocess=preprocess, post_process=post_process)
        if output_filename is None:
            pdf_name = Path(pdf_path).stem
            output_filename = f"{pdf_name}_cleaned.txt"
        if not output_filename.endswith('.txt'):
            output_filename += '.txt'
        output_path = os.path.join(self.output_dir, output_filename)
        if os.path.exists(output_path):
            base_name = output_filename.replace('.txt', '')
            counter = 1
            while os.path.exists(output_path):
                output_filename = f"{base_name}_{counter}.txt"
                output_path = os.path.join(self.output_dir, output_filename)
                counter += 1
            print(f"  Fichier existant, création de : {output_filename}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        lines_count = text.count('\n') + 1
        words_count = len(re.findall(r'\b\w+\b', text))
        print(f"\n✓ Fichier sauvegardé : {output_path}")
        print(f"  Caractères : {len(text)}")
        print(f"  Lignes : {lines_count}")
        print(f"  Mots : {words_count}")
        return output_path

# ... (la fonction process_pdf_for_rag reste inchangée) ...
def process_pdf_for_rag(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, output_dir: str = 'code/test_ocr') -> List[str]:
    """Traite un PDF et le découpe en chunks pour le RAG"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    processor = PDFOCRProcessor(lang='fra', dpi=300, output_dir=output_dir)
    # Note: ocr_pdf_to_txt a été modifié pour retourner le texte, pas le chemin
    text = processor.ocr_pdf(pdf_path, preprocess=True, post_process=True)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_text(text)
    return chunks
