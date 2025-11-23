import httpx
from openai import OpenAI

class LLM:
    def __init__(
        self,
        model: str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: str = "pas_de_clef",
    ):
        self.model_name = model
        self.base_url = base_url
        
        # 3. Initialisation du client OpenAI pointant vers ton serveur local
        # timeout=300.0 : On laisse 5 minutes max au modèle (sécurité pour les contextes longs)
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=300.0)

        # Définition de la personnalité de l'IA
        self.system_message = {
            "role": "system",
            "content": (
                "Tu es un assistant expert, pédagogique et précis. "
                "Tu aides les utilisateurs à partir de documents fournis. "
                "Réponds uniquement en te basant sur le contexte donné. "
                "Si la réponse n'est pas dans le contexte, dis-le clairement."
            ),
        }
        
        # 4. Vérification immédiate : est-ce que le serveur tourne ?
        self._check_connection()

    def _check_connection(self):
        """
        Vérifie si le serveur Llama.cpp est accessible dès le démarrage.
        """
        try:
            # Requête légère pour voir si le serveur répond
            self.client.models.list()
            print(f" [LLM] Connecté avec succès sur {self.base_url}")
            print(f"   Modèle cible : {self.model_name}")
        except httpx.ConnectError:
            # Message d'erreur pédagogique si le serveur est éteint
            print(f"\n [LLM] ERREUR : Impossible de se connecter à {self.base_url}")
            print("   -> Le serveur LLM semble éteint.")
            print("   -> Avez-vous lancé './start.sh' dans un autre terminal ?")
        except Exception as e:
            print(f" [LLM] Avertissement connexion : {e}")

    def infere(self, request: str) -> str:
        """
        Envoie le prompt (qui contient déjà le Contexte + la Question via rag.py) au LLM.
        """
        
        # 5. Mode "Stateless" (Sans mémoire) pour le RAG
        # On recrée la conversation à chaque appel.
        # Pourquoi ? Parce que rag.py envoie un énorme bloc de texte (le contexte).
        # Si on gardait l'historique, la 2ème question enverrait (Contexte1 + Q1 + R1 + Contexte2 + Q2)...
        # ... et ferait exploser la limite de mémoire du modèle (8192 tokens).
        messages = [
            self.system_message,
            {"role": "user", "content": request}
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  # Température basse = Réponses factuelles (pas d'invention)
                top_p=0.9,
                max_tokens=4096,  # Limite la longueur de la réponse
                stream=False
            )
            
            reply = completion.choices[0].message.content
            return reply

        except httpx.ConnectError:
            return "ERREUR CRITIQUE : Le serveur LLM est injoignable. Vérifiez './start.sh'."
        except Exception as e:
            return f"Une erreur est survenue pendant l'inférence du LLM : {e}"
        
    def reset_conversation(self):
        """
        Réinitialise l'historique de la conversation en conservant uniquement le message système.
        Utile pour éviter de dépasser la limite de tokens du contexte.
        """
        self.conversation = [self.system_message]
        print(" Conversation LLM réinitialisée")