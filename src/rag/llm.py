import numpy as np
from openai import OpenAI

class LLM :
    def __init__(self, 
                 model = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"  , 
                 base_url="http://127.0.0.1:8080/v1", 
                 api_key="pas_de_clef",

                 ) :
        

        self.client = OpenAI(
            base_url= base_url, 
            api_key= api_key,
            timeout=180.0  

        )
        self.LLAMA_MODEL = model 
        
        self.system_message = {"role": "system", "content": "You are a helpfull assistant. / no_think"}
        self.conversation = [self.system_message] # message system
        return

    def prompt(self):
        return
    
    def infere(self, request):
        
        
        
        self.conversation.append({"role": "user", "content": request}) # message user
        
        self. completion = self.client.chat.completions.create(
            model=self.LLAMA_MODEL,
            messages=self.conversation
            
        )   
        reply = self.completion.choices[0].message.content


            # On l'ajoute dans la conversation pour garder le contexte
        self.conversation.append({"role": "assistant", "content": reply})
        return reply 
    
    def reset_conversation(self):
        """
        Réinitialise l'historique de la conversation en conservant uniquement le message système.
        Utile pour éviter de dépasser la limite de tokens du contexte.
        """
        self.conversation = [self.system_message]
        print(" Conversation LLM réinitialisée")

