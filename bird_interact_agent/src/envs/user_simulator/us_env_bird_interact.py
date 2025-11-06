import openai
from openai import OpenAI
from src.envs.user_simulator.us_prompt import user_prompt_template
from src.envs.user_simulator.prompts import user_simulator_encoder, user_simulator_decoder
from typing import Dict, List, Tuple, Optional
from src.envs.bird_interact_env.token_counter import token_counter
from src.llm_utils.llm_provider import LLMProvider
import json
import os

USER_RESPONSE_CHARACTER_LIMIT = 400

# Set up logger
from rich.logging import RichHandler
import logging
import os

# Check if logging is disabled via environment variable
# DISABLE_LOGGING = os.getenv('DISABLE_USER_SIMULATOR_LOGGING', 'false').lower() == 'true'
DISABLE_LOGGING = True
if not DISABLE_LOGGING:
    handler = RichHandler(show_time=False)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
else:
    # Create a null logger that does nothing
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False


class UserSimulatorBirdInteractEnv:
    def __init__(self, llm_provider: LLMProvider, model_id: str, sql_env=None, use_encoder_decoder: bool = False, debug_mode: bool = False):
        self.llm_provider = llm_provider
        self.user_prompt = user_prompt_template  # A template string with placeholders
        self.encoder_prompt = user_simulator_encoder  # For ambiguity determination
        self.decoder_prompt = user_simulator_decoder  # For response generation
        self.model_id = model_id
        self.sql_env = sql_env  # SQL environment for test case evaluation
        self.ambiguous_query = ""
        self.clear_query = ""  # Original clear query
        self.clarifications = []  # List of dicts with ambiguity_term, sql_snippet, is_mask
        self.reference_sql = ""
        self.dialogue_history = []
        self.done = False
        self.logger = logger
        self.current_phase = 1  # 1 for first phase, 2 for follow-up phase
        self.follow_up_query = None
        self.follow_up_reference_sql = None
        # Define action costs (for reference, will be used by budget system in main code)
        self.action_costs = {
            "ask": 2,
            "submit": 3,
            "execute": 1,
            "get_schema": 1,
            "get_all_column_meanings": 1,
            "get_column_meaning": 0.5,
            "get_all_external_knowledge_names": 0.5,
            "get_knowledge_definition": 0.5,
            "get_all_knowledge_definitions": 1
        }
        self.phase_rewards = {
            1: 7,  # Phase 1 completion reward
            2: 3   # Phase 2 completion reward
        }
        self.total_reward = 0
        self.phase1_completed = False  # Flag to track phase 1 completion
        self.phase2_completed = False  # Flag to track phase 2 completion
        self.db_schema = ""
        self.segment_sql_cache = {}  # Cache for segmented SQL to avoid repeated computation
        self.use_encoder_decoder = use_encoder_decoder  # Whether to use encoder-decoder approach
        self.debug_mode = debug_mode  # Whether to print debug info
        self.info = {}
        # Reset token counter
        token_counter.reset()

    def set_sql_env(self, sql_env):
        """Set the SQL environment for test case evaluation"""
        self.sql_env = sql_env

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

    
    def segment_sql(self, sql: str) -> List[Tuple[str, str]]:
        """
        Segment SQL query into clauses. Uses cached results if available.
        
        Args:
            sql: SQL query to segment
            
        Returns:
            List of tuples with (clause_name, clause_text)
        """
        if sql in self.segment_sql_cache:
            return self.segment_sql_cache[sql]
        
        try:
            from src.envs.user_simulator.sql_parser import segment_sql
            segments = segment_sql(sql)
            self.segment_sql_cache[sql] = segments
            return segments
        except ImportError:
            self.logger.error("Could not import sql_parser module. Using simple segmentation.")
            # Simple segmentation as fallback
            segments = []
            parts = sql.split("\n")
            current_clause = "QUERY"
            current_text = ""
            
            for part in parts:
                stripped = part.strip().upper()
                if stripped.startswith("SELECT") or stripped.startswith("FROM") or \
                   stripped.startswith("WHERE") or stripped.startswith("GROUP BY") or \
                   stripped.startswith("ORDER BY") or stripped.startswith("HAVING"):
                    if current_text:
                        segments.append((current_clause, current_text))
                    current_clause = stripped.split()[0]
                    current_text = part
                else:
                    current_text += "\n" + part
            
            if current_text:
                segments.append((current_clause, current_text))
                
            self.segment_sql_cache[sql] = segments
            return segments
    
    def encode_ambiguity(self, question: str) -> str:
        """
        Determine ambiguity type based on agent's question
        
        Args:
            question: Agent's question to analyze
            
        Returns:
            Action representing the ambiguity type (labeled, unlabeled, or unanswerable)
        """
        # Prepare the encoder prompt
        prompt = self.encoder_prompt.replace('[[clarification_Q]]', question)
        
        # Format the user query ambiguity and knowledge ambiguity
        user_query_ambiguity = []
        knowledge_ambiguity = []
        
        if hasattr(self.sql_env, 'record') and self.sql_env.record:
            if 'user_query_ambiguity' in self.sql_env.record:
                user_query_ambiguity = self.sql_env.record['user_query_ambiguity']
            if 'knowledge_ambiguity' in self.sql_env.record:
                knowledge_ambiguity = self.sql_env.record['knowledge_ambiguity']
                
        ambiguity_json = {
            'user_query_ambiguity': user_query_ambiguity,
            'knowledge_ambiguity': knowledge_ambiguity
        }
        if self.current_phase == 1:
            prompt = prompt.replace('[[amb_json]]', json.dumps(ambiguity_json, indent=4))
        else:
            # phase-2 has no ambiguity
            prompt = prompt.replace('[[amb_json]]', json.dumps({}, indent=4))

        
        # Prepare SQL segments
        if isinstance(self.reference_sql, list):
            sql_list = self.reference_sql
        else:
            sql_list = [self.reference_sql]
            
        sql_segments = ""
        for i, sql in enumerate(sql_list):
            if i > 0:
                sql_segments += "\n===\n"
            for clause, text in self.segment_sql(sql):
                sql_segments += clause + ":\n" + text + "\n\n"
        
        prompt = prompt.replace('[[SQL_Glot]]', sql_segments.strip())
        prompt = prompt.replace('[[DB_schema]]', self.db_schema)
        
        # Count tokens
        token_counter.add_user_simulator_input(prompt)

        if self.debug_mode:
            self.logger.info(f"Encoder prompt: {prompt}")
        
        try:
            # Call the model using LLMProvider
            response = self.llm_provider.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            
            # Extract the response content
            content = response.strip()
            token_counter.add_user_simulator_output(content)
            logger.info(f"Raw response from encoder: {content}")
            # Extract action from response
            if "<s>" in content and "</s>" in content:
                action = content.split("<s>")[1].split("</s>")[0].strip()
            elif "</s>" in content:
                action = content.split("</s>")[0].strip()
            else:
                action = content.split("\n")[0].strip()
                
            self.logger.info(f"Encoded ambiguity: {action}")
            self.info["encoded_action"] = action
            return action
            
        except Exception as e:
            self.logger.error(f"Error in encode_ambiguity: {e}")
            return "unanswerable()"
    
    def decode_response(self, question: str, action: str) -> str:
        """
        Generate user response based on ambiguity action
        
        Args:
            question: Agent's question
            action: Ambiguity action from encoder
            
        Returns:
            User response
        """
        # Prepare the decoder prompt
        prompt = self.decoder_prompt.replace('[[clarification_Q]]', question)
        prompt = prompt.replace('[[Action]]', action)
        prompt = prompt.replace('[[clear_query]]', self.clear_query)
        
        # Format ambiguity JSON
        user_query_ambiguity = []
        knowledge_ambiguity = []
        
        if hasattr(self.sql_env, 'record') and self.sql_env.record:
            if 'user_query_ambiguity' in self.sql_env.record:
                user_query_ambiguity = self.sql_env.record['user_query_ambiguity']
            if 'knowledge_ambiguity' in self.sql_env.record:
                knowledge_ambiguity = self.sql_env.record['knowledge_ambiguity']
                
        ambiguity_json = {
            'user_query_ambiguity': user_query_ambiguity,
            'knowledge_ambiguity': knowledge_ambiguity
        }
        
        if self.current_phase == 1:
            prompt = prompt.replace('[[amb_json]]', json.dumps(ambiguity_json, indent=4))
        else:
            # phase-2 has no ambiguity
            prompt = prompt.replace('[[amb_json]]', json.dumps({}, indent=4))
        
        # Prepare SQL
        if isinstance(self.reference_sql, list):
            prompt = prompt.replace('[[GT_SQL]]', '\n'.join(self.reference_sql))
        else:
            prompt = prompt.replace('[[GT_SQL]]', self.reference_sql)
            
        # Prepare SQL segments
        sql_segments = ""
        if isinstance(self.reference_sql, list):
            sql_list = self.reference_sql
        else:
            sql_list = [self.reference_sql]
            
        for i, sql in enumerate(sql_list):
            if i > 0:
                sql_segments += "\n===\n"
            for clause, text in self.segment_sql(sql):
                sql_segments += clause + ":\n" + text + "\n\n"
        
        prompt = prompt.replace('[[SQL_Glot]]', sql_segments.strip())
        prompt = prompt.replace('[[DB_schema]]', self.db_schema)
        
        # Count tokens
        token_counter.add_user_simulator_input(prompt)

        if self.debug_mode:
            self.logger.info(f"Decoder prompt: {prompt}")
        
        try:
            # Call the model using LLMProvider
            response = self.llm_provider.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150
            )
            
            # Extract the response content
            content = response.strip()
            token_counter.add_user_simulator_output(content)
            logger.info(f"Raw response from decoder: {content}")
            # Extract response
            if "<s>" in content and "</s>" in content:
                user_response = content.split("<s>")[1].split("</s>")[0].strip()
            elif "</s>" in content:
                user_response = content.split("</s>")[0].strip()
            else:
                user_response = "I'm not sure I understand your question."
                
            self.logger.info(f"Decoded response: {user_response}")
            return user_response
            
        except Exception as e:
            self.logger.error(f"Error in decode_response: {e}")
            return "I'm not sure I understand your question."

    def reset(self, ambiguous_query: str, clarification_json: Dict, reference_sql: str, follow_up_query: Optional[str] = None, clear_query: Optional[str] = None, follow_up_reference_sql: Optional[str] = None):
        """
        Reset the user simulator with a new query and related information.
        
        Args:
            ambiguous_query: The initial ambiguous user query
            clarification_json: JSON containing clarification information, e.g. {"user_query_ambiguity": ..., "knowledge_ambiguity": ...}
            reference_sql: The reference SQL for evaluation
            follow_up_query: A follow-up query for phase 2 (if available)
            clear_query: The original clear query (for decoder)
        """
        # Reset token counter
        token_counter.reset()
        self.info = {}
        
        self.ambiguous_query = ambiguous_query
        self.clarifications = clarification_json
        self.reference_sql = reference_sql
        self.follow_up_query = follow_up_query
        self.follow_up_reference_sql = follow_up_reference_sql
        self.clear_query = clear_query if clear_query else ambiguous_query
        self.dialogue_history = [
            {"role": "system", "content": "You are a helpful user who will engage with an SQL agent."},
            {"role": "User (You)", "content": self.ambiguous_query}
        ]
        self.done = False
        self.current_phase = 1
        self.total_reward = 0
        self.phase1_completed = False
        self.phase2_completed = False
        self.segment_sql_cache = {}  # Clear SQL segment cache
        
        # Load DB schema from env
        self.db_schema = self.sql_env.get_schema()
        
        self.logger.info(f"User simulator reset with query: {ambiguous_query}")
        self.logger.info(f"Reference SQL: {reference_sql}")
        if follow_up_query:
            self.logger.info(f"Follow-up query: {follow_up_query}")
        if clear_query:
            self.logger.info(f"Clear query: {clear_query}")
        if self.use_encoder_decoder:
            self.logger.info("Using encoder-decoder approach for ambiguity handling")
        if follow_up_reference_sql:
            self.logger.info(f"Follow-up reference SQL: {follow_up_reference_sql}")
        
    def invoke_model(self) -> str:
        """
        Invoke the LLM to generate a user response based on dialogue history.
        
        Returns:
            str: Generated user response
        """
        # Prepare messages for the LLM
        messages = []
        for msg in self.dialogue_history:
            if "system" in msg["role"].lower():
                messages.append({"role": "system", "content": msg["content"]})
            elif "user" in msg["role"].lower():
                messages.append({"role": "user", "content": msg["content"]})
            else:
                messages.append({"role": "assistant", "content": msg["content"]})
                
        # Add information about ambiguities and clarifications
        system_message = """You are simulating a database user who asked the initial query.
Respond to the agent's questions based on the following information.
Keep responses brief (a few sentences at most) and focused on the question asked.
Provide information only when asked for it, without volunteering extra details."""
                
        if self.clarifications:
            system_message += "\n\nHere are the clarifications you can provide if asked about them:"
            system_message += json.dumps(self.clarifications, indent=2)
                
        # Replace the system message
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                messages[i]["content"] = system_message
                break
        
        # Make sure system message exists
        if not any(msg["role"] == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": system_message})
            
        # Count tokens for user simulator input
        input_text = " ".join([msg["content"] for msg in messages])
        token_counter.add_user_simulator_input(input_text)
        
        try:
            # Call the model using LLMProvider
            response = self.llm_provider.chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=150
            )
            
            # Extract the response content
            content = response.strip()
            
            # Count tokens for user simulator output
            token_counter.add_user_simulator_output(content)
            
            return content
        except Exception as e:
            self.logger.error(f"Error invoking model: {e}")
            error_response = "I don't understand. Could you clarify what you're asking?"
            token_counter.add_user_simulator_output(error_response)
            return error_response

    def step(self, action: str) -> Tuple[str, float, bool]:
        """
        Process an action from the agent and return the user's response.
        
        Args:
            action: The action from the agent, can be either:
                - ask(question): Ask the user a question
                - submit(sql): Submit SQL for testing
                
        Returns:
            Tuple[str, float, bool]: (response, reward, done)
        """
        if self.done:
            return "Task already completed.", self.total_reward, True

        # Count system input tokens
        token_counter.add_system_input(action)
        
        # Parse action
        if action.startswith("ask("):
            question = action[4:-1].strip().strip("'\"")
            self.dialogue_history.append({"role": "assistant", "content": question})
            
            if self.use_encoder_decoder:
                # Use the encoder-decoder approach 
                ambiguity_action = self.encode_ambiguity(question)
                response = self.decode_response(question, ambiguity_action)
            else:
                # Use the direct invoke approach
                response = self.invoke_model()
            
            self.dialogue_history.append(
                {"role": "User (You)", "content": response[:USER_RESPONSE_CHARACTER_LIMIT]}
            )
            self.logger.info(f"Agent Action: {action}")
            self.logger.info(f"User: {response}")
            
            # Return clean response
            token_counter.add_system_output(response)
            return response, 0, self.done
            
        elif action.startswith("submit("):
            sql = action[7:-1].strip().strip("'\"")
            self.dialogue_history.append({"role": "assistant", "content": f"Here's my SQL query: {sql}"})
            
            # Test the SQL
            if self.sql_env:
                passed, message = self.sql_env.run_test_case(sql)
            else:
                # Fallback if no SQL environment provided
                passed, message = False, "SQL environment not available for testing"
            
            if passed:
                if self.current_phase == 1 and self.follow_up_query and self.follow_up_reference_sql:
                    # Mark phase 1 as completed
                    self.phase1_completed = True
                    
                    # Award phase 1 reward
                    phase_reward = self.phase_rewards[1]
                    self.total_reward += phase_reward
                    
                    # Move to phase 2
                    self.current_phase = 2
                    self.ambiguous_query = self.follow_up_query
                    self.reference_sql = self.follow_up_reference_sql
                    self.dialogue_history.append({"role": "User (You)", "content": self.follow_up_query})
                    
                    # Prepare SQL environment for phase 2
                    if self.sql_env:
                        self.sql_env.start_phase_2()
                    
                    response = f"Your SQL is correct! (Reward: {phase_reward} points)\n\nNow, here's a follow-up question: {self.follow_up_query}"
                    token_counter.add_system_output(response)
                    self.logger.info(f"Phase 1 completed with reward: {phase_reward}")
                    
                    return response, phase_reward, self.done
                else:
                    # Task completed - either phase 1 or phase 2 is completed
                    self.done = True
                    phase_reward = self.phase_rewards[self.current_phase]
                    self.total_reward += phase_reward
                    
                    # Update phase completion flags
                    if self.current_phase == 1:
                        self.phase1_completed = True
                    elif self.current_phase == 2:
                        self.phase2_completed = True
                        # Phase 1 should already be completed, but ensure it's set
                        self.phase1_completed = True
                        
                    response = f"Your SQL is correct! Task completed.\n\nPhase {self.current_phase} completed with {phase_reward} points.\nTotal reward: {self.total_reward} points"
                    token_counter.add_system_output(response)
                    
                    # Log token usage summary at end of task
                    self.logger.info(token_counter.summary())
                    self.logger.info(f"Phase {self.current_phase} completed with reward: {phase_reward}")
                    self.logger.info(f"Phase 1 completed: {self.phase1_completed}, Phase 2 completed: {self.phase2_completed}")
                    self.logger.info(f"Total reward: {self.total_reward}")
                    
                    return response, self.total_reward, True
            else:
                # SQL failed, continue in current phase
                self.dialogue_history.append({"role": "User (You)", "content": f"Your SQL is not correct. {message} Please try again."})
                
                response = f"Your SQL is not correct. {message} Please try again."
                token_counter.add_system_output(response)
                return response, 0, self.done
        else:
            response = f"Invalid action format. Please use ask(question) or submit(sql)."
            token_counter.add_system_output(response)
            return response, 0, False