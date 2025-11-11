import gymnasium as gym
import numpy as np
import datetime, json, logging, os, re

from abc import ABC, abstractmethod
from rich.logging import RichHandler
from typing import Dict, List, Tuple

from src.utils import DataLoader

# Constants
AGENT_OBS = "agent_obs"
EVAL_OBS = "eval_obs"
CORRUPT_GOLD = "corrupt_gold"
ACTION_EXEC = "action_executed"
REWARD = "reward"

# Set up logger
handler = RichHandler(show_time=False)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

class BaseEnv(ABC, gym.Env):
    """Base class for BIRD-Interact environments"""
    name = "bird_interact"

    def __init__(self, **kwargs):
        """
        Initialize the environment.

        Args:
            **kwargs: Additional keyword arguments
                - data_path (str): Path to dataset
                - preprocess (function): Function to apply to environment before each episode
                - traj_dir (str): Directory to save trajectory files to
                - verbose (bool): Whether to print debug messages
        """
        super(BaseEnv, self).__init__()
        self.kwargs = kwargs
        self.logger = logger

        if "verbose" not in self.kwargs or self.kwargs["verbose"] != True:
            self.logger.disabled = True
        
        # Load dataset
        self.tool_mode = True
        if "data_path" in self.kwargs and self.kwargs["data_path"] is not None:
            self.data_path = self.kwargs["data_path"]
            self.data_loader = DataLoader(self.data_path)
            self.logger.info(f"Loaded dataset from {self.data_path}")
            self.tool_mode = False
        else:
            self.logger.info("No dataset provided, running in interactive mode")
        
        # Verify that preprocess function matches specifications
        self.preprocess = None
        if "preprocess" in self.kwargs and self.kwargs["preprocess"] is not None:
            self.logger.info("Verifying preprocess function...")
            preprocess = self.kwargs["preprocess"]
            assert(isinstance(preprocess, type(lambda x: x)))
            assert(preprocess.__annotations__["return"] == List)
            assert("record" in preprocess.__annotations__)
            assert(preprocess.__annotations__["record"] == Dict)
            self.preprocess = preprocess

        # Record logging directory if provided as a keyword argument
        self.traj_dir = None
        if "traj_dir" in self.kwargs and self.kwargs["traj_dir"]:
            self.traj_dir = self.kwargs["traj_dir"]

        self.logger.info("Environment Initialized")
        if not self.tool_mode:
            self.logger.info("* Note *: `reset` should be explicitly called to load new task episode")
    
    def step(self, action: str) -> Tuple[str, int, bool, Dict]:
        """
        Runs given action in environment and returns corresponding output
        
        Args:
            action (`str`) - command to run in bash shell
        
        Returns:
            observation (`str`) - standard output
            reward (`float`) - value between 0 and 1 quantifying correctness of output + environment state
            done (`bool`) - whether task is over
            info (`dict`) - additional information (e.g. debugging information)
        """
        if action == "skip":
            return "skipped", 0, True, {}
        if action.startswith("submit"):
            self.trajectory.append((action, None))
            reward, info = 0, {}
            if not self.tool_mode:
                reward, info = self.get_reward()
            if self.traj_dir is not None:
                self.save_trajectory()
            info[ACTION_EXEC] = True
            return self.observation, reward, True, info

        self.exec_action(action)
        self.logger.info(f"Action: {action}")
        self.logger.info(f"Observation: {self.observation}")
        self.trajectory.append((action, self.observation))
        return self.observation, 0, False, self.info

    def reset(self, index: int = None) -> Tuple[str, Dict]:
        """
        Create new session and reset environment variables

        Args:
            index (`int`) - index of query, gold pair to use for new session. If None, random index is used.
        """
        # Reset instance variables
        self.info = {}
        self.trajectory = []
        self.observation = None

        # Set query, gold command
        self.logger.info("-------------\nNew task episode initialized")
        self.query_idx = np.random.randint(0, len(self.data_loader)) if index is None else index
        self.record = self.data_loader.get(self.query_idx)
        self.query = self.record["query"]
        self.gold = self.record["gold"] if "gold" in self.record else "N/A"
        self.logger.info(f"Query: {self.query}")
        self.logger.info(f"Gold: {self.gold}")
        self.observation = self.query
        self.reward = None

        # Run preprocess function if provided
        if self.preprocess is not None:
            preprocess_cmds = self.preprocess(self.record)
            for cmd in preprocess_cmds:
                self.exec_action(cmd)
                if not self.info[ACTION_EXEC]:
                    raise RuntimeError(f"Preprocess command failed to execute successfully: {self.preprocess(self.record)}")
        
        return self.observation, self.info

    def save_trajectory(self):
        """
        Records trajectory of actions, observations, and rewards
        """        
        # Check if `logs` directory exists in current directory. If not, create it.
        self.traj_dir = os.path.dirname(os.path.abspath(__file__)) if self.traj_dir is None else self.traj_dir
        dir_path = os.path.join(self.traj_dir, f"logs_{self.name}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # Compile log information into a dictionary
        log_dict = {}
        log_dict["environment"] = self.name
        if not self.tool_mode:
            log_dict["query_idx"] = self.query_idx
            log_dict["query"] = self.query
            log_dict["gold"] = self.gold
            log_dict["reward"] = self.reward
        
        log_dict["trajectory"] = []
        for act, obs in self.trajectory:
            if isinstance(obs, List):
                obs = ''.join(map(str, obs))
            elif not isinstance(obs, str):
                obs = str(obs)
            log_dict["trajectory"].append({"action": act,  "observation": obs})
        
        log_dict["info"] = self.info

        # Create log file and write trajectory to it
        today = datetime.datetime.now().__str__()
        today = re.sub('[\.\-\:\s]+', '', today)[:-6]
        log_path = os.path.join(dir_path, f"log_{today}.json")
        try:
            with open(log_path, "w") as fp:
                json.dump(log_dict, fp, indent=2)
            self.logger.info(f"Saved trajectory to {log_path}")
        except Exception as e:
            self.logger.error(f"Failed to save trajectory to {log_path}: {e}")
    
    ###############################################
    # MARK: Methods to be implemented by subclass #
    ###############################################

    @abstractmethod
    def exec_action(self, action: str) -> None:
        """
        Executes given action in environment (called by `step` method)
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_reward(self) -> Tuple[float, Dict]:
        """
        Get reward value at current step of environment

        Returns:
            reward (`float`) - reward value
            info (`dict`) - additional information (e.g. debugging information)
        """
        raise NotImplementedError
    
    ########################################################
    # MARK: Optional methods to be implemented by subclass #
    ########################################################

    def close(self):
        """
        Clean up and close environment
        """
        pass
    
    # def reset_container(self) -> None:
    #     """
    #     Reset container to initial state
    #     """
    #     pass

    def get_available_actions(self) -> List:
        """
        Returns list of available actions in current environment state
        """
        pass