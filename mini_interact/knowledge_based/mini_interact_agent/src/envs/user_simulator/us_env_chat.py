import abc
import logging
from typing import Optional, List, Dict, Any
import json

import openai
from openai import OpenAI

from src.envs.user_simulator.us_prompt import user_prompt_system_template, AGENT_INIT_MESSAGE

# Set up logger
from rich.logging import RichHandler
import logging
handler = RichHandler(show_time=False)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class BaseUserSimulationEnv(abc.ABC):
    """Minimal version of your base environment interface."""
    metadata = {}

    @abc.abstractmethod
    def reset(self, instruction: Optional[str] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, content: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_cost(self) -> float:
        raise NotImplementedError


class UserSimulatorChatEnv(BaseUserSimulationEnv):
    """
    A user simulator that returns LLM responses as 'assistant' role,
    and receives the agent's messages as 'user' role.
    """

    MAX_CHAR_LIMIT = 400

    def __init__(self, client: OpenAI, model_id: str, env_id: int = 0, max_user_patience_turns: int = 4, disable_knowledge_ambiguity: bool = False):
        super().__init__()
        self.client = client
        self.model_id = model_id
        self.env_id = env_id
        # We store the entire conversation in self.messages:
        self.messages: List[Dict[str, str]] = []
        self.ambiguous_query = ""
        self.clarifications: List[Dict[str, Any]] = []
        self.reference_sql = ""
        self.db_schema = ""
        self.logger = logger
        self.total_cost = 0.0  # track usage if you want
        self.max_user_patience_turns = max_user_patience_turns
        # `disable_knowledge_ambiguity`:
            # Default is False, meaning we consider the knowledge ambiguity.
            # If True,
            # TODO: Discuss with team.
            # (1) the user don't consider the knowledge ambiguity when calculate the ambiguity resolution reward,
            # (2) the clarification of knowledge ambiguity will not be shown to the user, in case of leaking the clarification.
        self.disable_knowledge_ambiguity = disable_knowledge_ambiguity  

    def reset(self, user_info_dict: Dict[str, Any]) -> str:
        """
        Reset environment, parse the instruction (if any), and start fresh.
        Return the initial user simulator message (which will be in 'assistant' role).
        """
        self.total_cost = 0.0
        self.messages.clear()

        # Parse JSON instruction if provided
        # e.g. {
        #   "ambiguous_query": "...",
        #   "reference_sql": "...",
        #   "clarifications": [...]
        # }     
        if user_info_dict:
            try:
                self.ambiguous_query = user_info_dict.get("query", "")
                self.reference_sql = user_info_dict.get("gold", "")
                self.clarifications = user_info_dict["ambiguity_terms"]["ambiguity_term"]
                if not self.disable_knowledge_ambiguity:    
                    self.clarifications += user_info_dict["ambiguity_terms"]["ambiguity_knowledge"]
                clarifications_str = self._format_clarifications(self.clarifications)
                self.db_schema = user_info_dict.get("db_schema", "")
                self.golden_kb = user_info_dict.get("golden_kb", "")
            except Exception as e:
                self.logger.warning(f"Failed to parse 'user_info_dict' JSON: {e}")

        self.logger.info("=== NEW SIMULATION ===")
        self.logger.info(f"Ambiguous Query: {self.ambiguous_query}")
        self.logger.info(f"Reference SQL: {self.reference_sql}")
        self.logger.info(f"Clarifications: {clarifications_str}")

        # 1) Add a system message instructing the LLM to act as the user:
        system_prompt = user_prompt_system_template.format(
            db_name=self.db_schema["db"],
            db_schema=self.db_schema["schema"],
            ambiguous_query=self.ambiguous_query,
            reference_sql=self.reference_sql,
            clarifications=clarifications_str,
            golden_kb=self.golden_kb,
        )
        self.messages.append({
            "role": "system",
            "content": system_prompt
        })

        # Add a agent's asking question and the initial user simulator's query
        self.messages.append(
            {
            "role": "user", # here, "user" is the api's perspective, not the user simulator's role
            "content": AGENT_INIT_MESSAGE   # the initial agent's message, e.g. "Hi! How can I help you today?"
            }
        )
        self.messages.append(
            {
            "role": "assistant", 
            "content": self.ambiguous_query
            }
        )
        return self.ambiguous_query

    def step(self, content: str) -> str:
        """
        The 'agent' calls this with its message `content`. We store it as role='user',
        then we ask the LLM as the user simulator to produce the next 'assistant' response.
        """
        # 1) Append the incoming message as 'user'
        self.messages.append({
            "role": "user",
            "content": content
        })
        # 2) Generate the next "assistant" response
        us_reply = self._generate_assistant_reply()

        # Calculate ambiguity resolution reward
        policy_messages = [item["content"] for item in self.messages if item["role"] == "user"]
        amb_res_reward = self.get_amb_res_reward(policy_messages)
        return us_reply, amb_res_reward

    def _format_clarifications(self, clarification_jsons: List[Dict[str, Any]]) -> str:
        # [{"ambiguity_term": "some", "sql_snippet": "LIMIT 2", "ambiguity_type": "Lexical Ambiguity", "is_mask": false}, ...]
        # format as:
        # > Potential ambiguities terms in your query and key SQL snippets as clarification:
        # > - 'some': LIMIT 2 (Lexical Ambiguity)
        # ...

        formatted_clarifications = []
        for clarification_json in clarification_jsons:
            formatted_clarifications.append(f"- '{clarification_json['ambiguity_term']}':\t```sql {repr(clarification_json['sql_snippet'])}```")
        return "\n".join(formatted_clarifications)
        
    def _generate_assistant_reply(self) -> str:
        """
        Invokes the LLM with self.messages, expecting the next role='assistant'.
        """
        # We'll simply pass self.messages to the chat completion.
        # The LLM will see system instructions + all prior user/assistant messages,
        # and produce a new 'assistant' response that simulates the user.

        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=self.messages,
                max_tokens=300,
                temperature=0.0,
            )

            # The LLM's next message (role=assistant)
            reply_content = completion.choices[0].message.content.strip()
        except Exception as e:

            self.logger.warning(f"OpenAI Error: {e}")
            reply_content = "I'm not sure how to respond."

        # Enforce the character limit
        reply_content = reply_content[: self.MAX_CHAR_LIMIT]

        # 3) Append the new message with role='assistant'
        self.messages.append({
            "role": "assistant",
            "content": reply_content
        })

        # self.logger.info(f"[User Simulator] {reply_content}")
        return reply_content

    def get_amb_res_reward(self, policy_actions: List[str]) -> float:
        """
        Calculate the reward based on the ratio of the agent's question that hit the ambiguity terms.
        reward = num_amb_hit / num_amb_total
        """
        ambiguity_terms = []
        for clarification in self.clarifications:
            ambiguity_terms.append(clarification["ambiguity_term"])
        num_amb_total = len(ambiguity_terms)
        policy_actions_str = "\n".join(policy_actions)
        num_amb_hit = sum([1 for item in ambiguity_terms if item in policy_actions_str])
        return num_amb_hit / num_amb_total

    def get_total_cost(self) -> float:
        return self.total_cost



if __name__ == "__main__":
    # 1) Instantiate environment
    my_openai_client = openai.OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    )   
    env = UserSimulatorChatEnv(client=my_openai_client, model_id="gemma3:12b")

    # 2) Build sample instruction JSON
    sample_instruction = {
        "ambiguous_query": "I want to select all customers from the database.",
        "reference_sql": "SELECT * FROM db1.customers",
        "ambiguity_terms": {
            "ambiguity_term": [
                {"ambiguity_term": "customers", "sql_snippet": "db1.customers", "ambiguity_type": "Lexical Ambiguity", "is_mask": False}
            ],
            "ambiguity_knowledge": [
                {"ambiguity_term": "customers", "sql_snippet": "db2.customers", "ambiguity_type": "Lexical Ambiguity", "is_mask": False}
            ]
        }
    }

    # 3) Reset environment => get initial "assistant" message from the user simulator
    first_usr_msg = env.reset(json.dumps(sample_instruction))
    # print("[UserSim - assistant]:", first_usr_msg)

    # 4) Now the 'agent' sends a message
    agent_text = "Which schema should I use for 'customers'?"
    # print("[Agent]:", agent_text)
    reply = env.step(agent_text)
    # print("[UserSim - assistant]:", reply)

    # 6) Check cost
    # print("Total cost:", env.get_total_cost())


