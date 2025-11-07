import sys
from pathlib import Path

# ⚠️ AJOUTER CES 3 LIGNES EN PREMIER (avant tous les autres imports)
ROOT_DIR = Path(__file__).parent.parent  # Remonte de launchers/ à RAG_CGT/
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.rag import Rag

# Configuration de la page
st.set_page_config(
    page_title="Assistant RAG CGT",
    page_icon="🤖",
    layout="wide"
)

# Initialisation du système RAG (avec cache)
@st.cache_resource
def init_rag():
    """Initialise le système RAG une seule fois"""
    return Rag()

# Interface principale
def main():
    st.title(" Assistant RAG - Base documentaire CGT")
    st.markdown("---")
    
    # Initialisation
    try:
        rag_instance = init_rag()
    except Exception as e:
        st.error(f" Erreur d'initialisation : {e}")
        return
    
    # Sidebar : Sélection de la collection
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Lister les collections disponibles
        collections = rag_instance.retrival.chroma_storage.list_collection_names()
        
        if not collections:
            st.warning(" Aucune collection disponible")
            st.info("Utilisez `manage_collections.py` pour créer une collection")
            return
        
        # Menu déroulant pour sélectionner la collection
        selected_collection = st.selectbox(
            "Collection :",
            collections,
            index=0
        )
        
        # Basculer vers la collection sélectionnée
        if selected_collection:
            rag_instance.retrival.chroma_storage.switch_collection(selected_collection)
            
            # Afficher les stats de la collection
            stats = rag_instance.retrival.get_stats()
            
            st.metric(" Documents", stats['total_documents'])
            st.metric(" Fichiers sources", stats['total_fichiers'])
            
            # Métadonnées de la collection
            collection = rag_instance.retrival.chroma_storage.collection
            metadata = collection.metadata
            
            with st.expander(" Détails de la collection"):
                if metadata.get("chunk_size"):
                    st.write(f"**Taille chunks :** {metadata.get('chunk_size')} caractères")
                if metadata.get("overlap"):
                    st.write(f"**Overlap :** {metadata.get('overlap')} caractères")
                if metadata.get("model"):
                    st.write(f"**Modèle :** {metadata.get('model')}")
                if metadata.get("created_at"):
                    st.write(f"**Créée le :** {metadata.get('created_at')[:10]}")
    
    # Zone principale : Question-Réponse
    st.header(" Posez votre question")
    
    # Champ de saisie de la question
    question = st.text_area(
        "Votre question :",
        placeholder="Ex: Quels sont les droits des agents ?",
        height=100
    )
    
    # Bouton pour soumettre la question
    col1, col2 = st.columns([1, 4])
    
    with col1:
        submit_button = st.button(" Rechercher", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button(" Effacer", use_container_width=True)
    
    # Gestion du bouton Effacer
    if clear_button:
        st.rerun()
    
    # Gestion de la soumission
    if submit_button and question:
        with st.spinner(" Recherche en cours..."):
            try:
                # Appel du système RAG
                reponse = rag_instance.respond(question)
                
                # Affichage de la réponse
                st.markdown("---")
                st.subheader(" Réponse")
                
                # Séparer la réponse des sources
                if "Sources (Top-" in reponse:
                    reponse_text, sources_text = reponse.split("\n\nSources (Top-", 1)
                    sources_text = "Sources (Top-" + sources_text
                else:
                    reponse_text = reponse
                    sources_text = None
                
                # Afficher la réponse
                st.markdown(reponse_text)
                
                # Afficher les sources dans un expander
                if sources_text:
                    with st.expander(" Voir les sources", expanded=True):
                        st.text(sources_text)
                
            except Exception as e:
                st.error(f" Erreur lors de la génération de la réponse : {e}")
    
    elif submit_button and not question:
        st.warning(" Veuillez entrer une question")
    
    # Section historique (optionnel)
    with st.sidebar:
        st.markdown("---")
        
        #  Réinitialiser la session LLM
        if st.button(" Réinitialiser la session LLM"):
            try:
                rag_instance.llm.reset_conversation()
                st.success(" Session LLM réinitialisée ! L'historique des conversations a été effacé.")
                st.info(" La mémoire du modèle est maintenant vide, vous pouvez poser de nouvelles questions sans risque de dépassement de tokens.")
            except Exception as e:
                st.error(f" Erreur lors de la réinitialisation : {e}")
        
        if st.button(" Recharger l'application"):
            st.cache_resource.clear()
            st.rerun()
            
if __name__ == "__main__":
    main()
