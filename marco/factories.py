from typing import Dict, Type, Any, Optional
from abc import ABC, abstractmethod

from marco.agents.base import Agent
from marco.agents import (
    Planner, Analyst, Solver, Reflector
)

class AgentFactory(ABC):
    @abstractmethod
    def create_agent(self, agent_type: str, config: Dict[str, Any], **kwargs) -> Agent:
        pass

class DefaultAgentFactory(AgentFactory):
    
    def __init__(self):
        self._agent_registry: Dict[str, Type[Agent]] = {
            'Analyst': Analyst,
            'Reflector': Reflector,
            'Planner': Planner,
            'Solver': Solver,
        }
    
    def register_agent(self, name: str, agent_class: Type[Agent]) -> None:
        self._agent_registry[name] = agent_class
    
    def create_agent(self, agent_type: str, config: Dict[str, Any], **kwargs) -> Agent:
        if agent_type not in self._agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = self._agent_registry[agent_type]
        return agent_class(**config, **kwargs)
    
    def get_available_agents(self) -> Dict[str, Type[Agent]]:
        return self._agent_registry.copy()

class ToolProvider:
    
    def __init__(self, tool_configs: Dict[str, Dict[str, Any]]):
        self.tool_configs = tool_configs
        self._tool_cache = {}
    
    def get_tool_config(self, tool_name: str) -> Optional[Dict[str, Any]]:
        return self.tool_configs.get(tool_name)
    
    def get_all_tool_configs(self) -> Dict[str, Dict[str, Any]]:
        return self.tool_configs.copy()

class ConfigManager:
    
    def __init__(self, config_data: Dict[str, Any]):
        self.config_data = config_data
    
    def get_system_config(self) -> Dict[str, Any]:
        return {
            'supported_tasks': self.config_data.get('supported_tasks', []),
            'max_step': self.config_data.get('max_step', 10),
            'agent_prompt': self.config_data.get('agent_prompt'),
            'data_prompt': self.config_data.get('data_prompt'),
        }
    
    def get_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        return self.config_data.get('agents', {})
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        return self.config_data.get('agents', {}).get(agent_name, {})
