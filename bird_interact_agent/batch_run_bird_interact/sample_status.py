from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class SampleStatus:
    """Holds the status and interaction history for a single sample."""
    idx: int
    original_data: Dict[str, Any]
    current_prompt: str = ""
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    remaining_budget: float = 0.0
    total_budget: float = 0.0
    phase1_completed: bool = False
    phase2_completed: bool = False
    task_finished: bool = False
    current_turn: int = 0
    current_phase: int = 1 # 1 or 2
    # Fields to store temporary results between steps
    last_agent_response: Optional[str] = None
    parsed_action_object: Optional[str] = None
    parsed_action: Optional[str] = None
    parsed_thought: Optional[str] = None
    last_observation: Optional[str] = None
    last_reward: Optional[float] = None
    last_user_response: Optional[str] = None
    force_submit: bool = False # Flag if budget runs out
    successful_phase1_sql: Optional[str] = None # Added field

    # Add fields for budget tracking categories if needed
    # env_interact_used: float = 0.0
    # submit_used: float = 0.0
    # user_patience_used: float = 0.0

    # You might add methods here for updating status, budget, etc.
    def add_turn_log(self, thought: str, interaction_object: str, action: str, observation: str, reward: float, budget_info: Dict):
        """Adds a log entry for the completed turn."""
        self.interaction_history.append({
            "turn": self.current_turn,
            "phase": self.current_phase,
            "thought": thought,
            "interaction_object": interaction_object,
            "action": action,
            "observation": observation,
            "reward": reward, # Reward *received* in this turn (usually 0 unless it's the final submit)
            "budget_after_action": budget_info
        })

    def get_full_interaction_prompt(self) -> str:
        """Constructs the full prompt history for the agent."""
        prompt = self.current_prompt # Initial query + budget info
        for turn_log in self.interaction_history:
            prompt += f"""<thought>{turn_log['thought']}</thought>
<interaction_object>{turn_log['interaction_object']}</interaction_object>
<action>{turn_log['action']}</action>

Observation: {turn_log['observation']}

"""
        return prompt 