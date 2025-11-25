from typing import Dict, Any, Optional, Set
from abc import ABC, abstractmethod
from loguru import logger

from marco.agents.base import Agent
from marco.factories import AgentFactory, ConfigManager

class SystemState:
    
    def __init__(self):
        self.step_count = 0
        self.max_steps = 10
        self.finished = False
        self.result = None
        self.chat_history = []
        self.action_history = []
        self.analyzed_items: Set[str] = set()
        self.analyzed_users: Set[str] = set()
        self.execution_context = {}
    
    def reset(self):
        self.step_count = 0
        self.finished = False
        self.result = None
        self.chat_history = []
        self.action_history = []
        self.analyzed_items.clear()
        self.analyzed_users.clear()
        self.execution_context.clear()
    
    def increment_step(self):
        self.step_count += 1
    
    def is_max_steps_reached(self) -> bool:
        return self.step_count >= self.max_steps
    
    def add_to_history(self, action: str, result: Any = None):
        self.action_history.append({
            'step': self.step_count,
            'action': action,
            'result': result
        })

class AgentCoordinator:
    
    def __init__(self, agent_factory: AgentFactory):
        self.agent_factory = agent_factory
        self.agents: Dict[str, Agent] = {}
    
    def initialize_agents(self, agent_configs: Dict[str, Dict[str, Any]], **kwargs):
        import json
        import os
        from loguru import logger
        
        self.agents.clear()
        
        system = kwargs.get('system')
        logger.debug(f"🔍 initialize_agents: system={system is not None}")
        model_override = getattr(system, 'model_override', None) if system else None
        
        for agent_name, config in agent_configs.items():
            try:
                # Apply model override if specified
                final_agent_config = config.copy()
                
                dataset = kwargs.get('dataset') or getattr(system, 'dataset', None)
                task = kwargs.get('task') or getattr(system, 'task', None)
                
                if dataset and 'config_path' in final_agent_config:
                    import json
                    with open(final_agent_config['config_path'], 'r') as f:
                        agent_config_content = f.read()
                    
                    if dataset:
                        agent_config_content = agent_config_content.replace('{dataset}', dataset)
                    if task:
                        agent_config_content = agent_config_content.replace('{task}', task)
                    
                    substituted_config = json.loads(agent_config_content)
                    final_agent_config['config'] = substituted_config
                
                if model_override and system:
                    if 'config_path' in config:
                        with open(config['config_path'], 'r') as f:
                            agent_llm_config = json.load(f)
                        
                        agent_llm_config = system._apply_model_override(agent_llm_config)
                        final_agent_config['config'] = agent_llm_config
                
                agent = self.agent_factory.create_agent(agent_name, final_agent_config, **kwargs)
                self.agents[agent_name] = agent
                logger.info(f"Initialized agent: {agent_name}")
            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_name}: {e}")
                raise
        
        system = kwargs.get('system')
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        return self.agents.get(agent_name)
    
    def reset_all_agents(self):
        for agent in self.agents.values():
            agent.reset()
    
    def execute_agent_action(self, agent_name: str, action: str, **kwargs) -> Any:
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent {agent_name} not found")
        
        try:
            if hasattr(agent, action):
                method = getattr(agent, action)
                return method(**kwargs)
            else:
                raise AttributeError(f"Agent {agent_name} has no action {action}")
        except Exception as e:
            logger.error(f"Error executing {action} on {agent_name}: {e}")
            raise

class SystemOrchestrator(ABC):
    
    def __init__(self, config_manager: ConfigManager, agent_coordinator: AgentCoordinator):
        self.config_manager = config_manager
        self.agent_coordinator = agent_coordinator
        self.state = SystemState()
    
    @abstractmethod
    def execute_workflow(self, **kwargs) -> Any:
        pass
    
    def initialize(self, **kwargs):
        system_config = self.config_manager.get_system_config()
        self.state.max_steps = system_config.get('max_step', 10)
        
        agent_configs = self.config_manager.get_agent_configs()
        self.agent_coordinator.initialize_agents(agent_configs, **kwargs)
    
    def reset(self):
        self.state.reset()
        self.agent_coordinator.reset_all_agents()
    
    def is_finished(self) -> bool:
        return self.state.finished or self.state.is_max_steps_reached()

class MARCOOrchestrator(SystemOrchestrator):
    
    def execute_workflow(self, **kwargs) -> Any:
        self.state.reset()
        
        try:
            # Phase 1: Planning
            plan = self._execute_planning_phase(**kwargs)
            if not plan:
                return None
            
            # Phase 2: Working
            worker_results = self._execute_working_phase(plan, **kwargs)
            
            # Phase 3: Solving
            solution = self._execute_solving_phase(plan, worker_results, **kwargs)
            
            self.state.finished = True
            self.state.result = solution
            
        except Exception as e:
            logger.error(f"Error in MARCO workflow: {e}")
            
        return self.state.result
    
    def _execute_planning_phase(self, **kwargs) -> Optional[str]:
        planner = self.agent_coordinator.get_agent('Planner')
        if not planner:
            logger.error("Planner agent not found")
            return None
        
        return self.agent_coordinator.execute_agent_action('Planner', 'forward', **kwargs)
    
    def _execute_working_phase(self, plan: str, **kwargs) -> Dict[str, Any]:
        worker_results = {}
        
        # Parse plan to identify required workers
        # This is a simplified implementation
        workers = ['Analyst']
        
        for worker_name in workers:
            worker = self.agent_coordinator.get_agent(worker_name)
            if worker:
                try:
                    result = self.agent_coordinator.execute_agent_action(
                        worker_name, 'forward', plan=plan, **kwargs
                    )
                    worker_results[worker_name] = result
                except Exception as e:
                    logger.warning(f"Worker {worker_name} failed: {e}")
        
        return worker_results
    
    def _execute_solving_phase(self, plan: str, worker_results: Dict[str, Any], **kwargs) -> Optional[str]:
        solver = self.agent_coordinator.get_agent('Solver')
        if not solver:
            logger.error("Solver agent not found")
            return None
        
        return self.agent_coordinator.execute_agent_action(
            'Solver', 'forward', plan=plan, worker_results=worker_results, **kwargs
        )
