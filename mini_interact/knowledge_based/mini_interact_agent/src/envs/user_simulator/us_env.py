import openai
from openai import OpenAI
from src.envs.user_simulator.us_prompt import user_prompt_template

USER_RESPONSE_CHARACTER_LIMIT = 400

# Set up logger
from rich.logging import RichHandler
import logging
handler = RichHandler(show_time=False)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class UserSimulatorEnv:
    def __init__(self, client: OpenAI, model_id, env_id=0):
        self.client: OpenAI = client
        self.user_prompt = user_prompt_template  # A template string with placeholders
        self.model_id = model_id
        self.env_id = env_id
        self.ambiguous_query = ""
        self.clarifications = []  # List of dicts with ambiguity_term, sql_snippet, is_mask
        self.reference_sql = ""
        self.dialogue_history = []
        self.done = False
        self.logger = logger

    def get_dialogue_history(self):
        return [{"role": d["role"], "content": d["content"]} for d in self.dialogue_history]

    def str_dialogue_history(self):
        result = ""
        round_id = 1
        for i, d in enumerate(self.dialogue_history):
            if i % 2 == 0:
                result += f"--- Round {round_id} ---\n{d['role']}:\n{d['content']}\n\n"
                round_id += 1
            else:
                result += f"{d['role']}:\n{d['content']}\n\n"
        return result

    def reset(self, ambiguous_query, clarification_json, reference_sql):
        self.ambiguous_query = str(ambiguous_query)
        self.reference_sql = reference_sql
        
        # Flatten the clarification JSON
        self.clarifications = []
        for key in ["ambiguity_term", "ambiguity_knowledge"]:
            if key in clarification_json:
                for item in clarification_json[key]:
                    self.clarifications.append({
                        "ambiguity_term": item["ambiguity_term"],
                        "sql_snippet": item["sql_snippet"],
                        "is_mask": item["is_mask"]
                    })
        
        self.done = False
        self.dialogue_history = []
        self.dialogue_history.append({"role": "User (You)", "content": self.ambiguous_query})
        self.logger.info("-------------\nNew user simulator initialized")
        self.logger.info(f"Ambiguous Query: {self.ambiguous_query}")
        self.logger.info(f"Reference SQL: {self.reference_sql}")
        self.logger.info(f"Clarifications: {self.clarifications}")

        return self.get_dialogue_history()

    def invoke_model(self):
        # Format clarifications into a readable string
        clarifications_str = "\n".join(
            [f"- '{c['ambiguity_term']}': {c['sql_snippet']}" 
             for c in self.clarifications]
        )
        
        # Construct the prompt using the template
        prompt = self.user_prompt.format(
            ambiguous_query=self.ambiguous_query,
            clarifications=clarifications_str if clarifications_str else "No specific clarifications provided.",
            reference_sql=self.reference_sql,
            dialogue_history=self.str_dialogue_history()
        )
        self.logger.info(f"Prompt: {prompt}")
        # Call the model
        for _ in range(3):
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful user responding to clarification questions."},
                    {"role": "user", "content": prompt},
                ]
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0,
                )
                return completion.choices[0].message.content.strip()
            except openai.BadRequestError:
                return "I'm not sure how to respond."
        return "No response."

    def step(self, action):
        if self.done:
            return None

        question = action.strip()
        self.dialogue_history.append({"role": "assistant", "content": question})

        response = self.invoke_model()
        self.dialogue_history.append(
            {"role": "User (You)", "content": response[:USER_RESPONSE_CHARACTER_LIMIT]}
        )
        self.logger.info(f"Agent Action: {action}")
        self.logger.info(f"User: {response}")
        return response