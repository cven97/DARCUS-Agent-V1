class SpecialtyAgent:
    def __init__(self, name, system_prompt, role):
        self.name = name
        self.system_prompt = system_prompt
        self.role = role

    def get_system_prompt(self, config):
        # Permet de surcharger le prompt via la configuration globale
        config_key = f"{self.name.replace('-', '_')}"
        return config.get(config_key, {}).get("system_prompt", self.system_prompt)


class ConversationAgent(SpecialtyAgent):
    def __init__(self):
        super().__init__(
            name="conversation-agent",
            role="Conversation & Assistance générale",
            system_prompt=(
                "Tu es un expert en communication.\n\n"
                "Tu privilégies :\n"
                "- clarté\n"
                "- précision\n"
                "- concision\n"
                "- pédagogie\n\n"
                "Tu réponds toujours dans la langue de l'utilisateur."
            )
        )


class CodingAgent(SpecialtyAgent):
    def __init__(self):
        super().__init__(
            name="coding-agent",
            role="Développement logiciel & Architecture",
            system_prompt=(
                "Tu es un architecte logiciel senior.\n\n"
                "Priorités :\n"
                "1. Exactitude\n"
                "2. Performance\n"
                "3. Sécurité\n"
                "4. Maintenabilité\n\n"
                "Toujours expliquer tes choix techniques de manière claire."
            )
        )


class ResearchAgent(SpecialtyAgent):
    def __init__(self):
        super().__init__(
            name="research-agent",
            role="Recherche d'informations & Web",
            system_prompt=(
                "Tu es un agent de recherche expert.\n"
                "Ta tâche est de collecter de l'information précise et fiable sur le web ou dans les fichiers locaux.\n"
                "Utilise les outils à ta disposition pour chercher, extraire et synthétiser les données.\n"
                "Cite toujours tes sources et évite toute spéculation."
            )
        )


class CriticAgent(SpecialtyAgent):
    def __init__(self):
        super().__init__(
            name="critic-agent",
            role="Audit de réponses & de code",
            system_prompt=(
                "Tu es un auditeur et critique expert.\n"
                "Ta tâche est de relire et d'évaluer de manière critique le code généré ou les réponses rédigées par d'autres agents.\n"
                "Vérifie la cohérence, la précision, les bugs potentiels, les failles de sécurité et les hallucinations.\n"
                "Fournis des corrections claires et constructives si nécessaire."
            )
        )
