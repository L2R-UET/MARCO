import json
from typing import Any, Dict, Optional, TYPE_CHECKING
from loguru import logger

from marco.systems.base import System
from marco.factories import DefaultAgentFactory
from marco.components import AgentCoordinator
from marco.utils import duration_tracker, read_json
import json

if TYPE_CHECKING:
    from marco.agents import Analyst, Reflector, Solver

class MARCOPSystem(System):
    """
    MARCO-P (MARCO without Planner) System for Ablation Studies.
    
    This system implements a simplified MARCO pattern without the planning phase:
    1. Analysis: Analyst directly analyzes user and candidate items
    2. Solving: Solver synthesizes analyst insights into final recommendations
    
    This is designed for ablation testing to evaluate the contribution of the Planner.
    """
    
    def __init__(self, task: str, config_path: str, leak: bool = False, 
                 dataset: Optional[str] = None, enable_reflection_rerun: bool = False, *args, **kwargs) -> None:
        self.agent_factory = DefaultAgentFactory()
        self.agent_coordinator = AgentCoordinator(self.agent_factory)
        self.enable_reflection_rerun = enable_reflection_rerun
        
        super().__init__(task, config_path, leak, dataset, *args, **kwargs)
        
    def init(self, *args, **kwargs) -> None:
        self.max_step: int = self.config.get('max_step', 10)
        assert 'agents' in self.config, 'Agents are required.'
        
        agents_config = self.config.get('agents', {})
        final_agents_config = {}
        
        for agent_name, agent_spec in agents_config.items():
            config_path = agent_spec.get('config_path')
            prompt_config = agent_spec.get('prompt_config')
            
            # Read agent LLM config
            agent_llm_config = read_json(config_path)
            
            # Apply model override if specified
            if hasattr(self, 'model_override') and self.model_override:
                agent_llm_config = self._apply_model_override(agent_llm_config)
            
            # Prepare final agent config for coordinator
            final_agent_config = {
                'config_path': config_path,
                'prompt_config': prompt_config,
                'config': agent_llm_config
            }
            final_agents_config[agent_name] = final_agent_config
        
        self.agent_coordinator.initialize_agents(final_agents_config, **self.agent_kwargs)
        
        self.manager_kwargs = {
            'max_step': self.max_step,
            'task_type': self.task,
        }
        
        self.execution_results = {}
        self.phase = 'analysis'
        
        # Track reflection improvements
        self.reflection_improvements = []
        self.reflection_all_reruns = []
        self.total_reflections_triggered = 0
        
        # Sample tracking
        self._current_sample_idx = -1
        self._current_user_id = -1

    @staticmethod
    def supported_tasks() -> list[str]:
        return ['rp', 'sr', 'gen', 'chat']

    @property
    def analyst(self) -> Optional['Analyst']:
        return self.agent_coordinator.get_agent('Analyst')

    @property
    def solver(self) -> Optional['Solver']:
        return self.agent_coordinator.get_agent('Solver')

    @property
    def reflector(self) -> Optional['Reflector']:
        return self.agent_coordinator.get_agent('Reflector')
    
    def set_data(self, input: str, context: str, gt_answer: Any, data_sample: Optional[Any] = None) -> None:
        super().set_data(input, context, gt_answer, data_sample)
        
        if data_sample is not None:
            self._current_user_id = data_sample.get('user_id', -1) if hasattr(data_sample, 'get') else (data_sample['user_id'] if 'user_id' in data_sample else -1)
        else:
            self._current_user_id = -1

    def reset(self, clear: bool = False, preserve_progress: bool = False, *args, **kwargs) -> None:
        saved_sample_idx = getattr(self, '_current_sample_idx', -1)
        saved_user_id = getattr(self, '_current_user_id', -1)
        
        super().reset(clear, *args, **kwargs)
        
        if not preserve_progress:
            self.execution_results.clear()
            if hasattr(self, '_last_solution'):
                delattr(self, '_last_solution')
            if hasattr(self, '_last_final_answer'):
                delattr(self, '_last_final_answer')
        else:
            self._current_sample_idx = saved_sample_idx
            self._current_user_id = saved_user_id
        
        self.agent_coordinator.reset_all_agents()
        
        self.phase = 'analysis'

    def forward(self, user_input: Optional[str] = None, reset: bool = True) -> Any:
        try:
            # Prepare data integration
            if self.task == 'chat':
                self.manager_kwargs['history'] = self.chat_history if hasattr(self, 'chat_history') else []
            else:
                self.manager_kwargs['input'] = self.input
                self.manager_kwargs['data_sample'] = self.data_sample
                
            if reset:
                self.reset()
                
            if self.task == 'chat':
                assert user_input is not None, 'User input is required for chat task.'
                if hasattr(self, 'add_chat_history'):
                    self.add_chat_history(user_input, role='user')
            
            # Check if required agents are available
            if self.analyst is None or self.solver is None:
                logger.error("MARCO-P requires Analyst and Solver agents.")
                raise ValueError("MARCO-P requires Analyst and Solver agents.")
            
            result = self._execute_workflow()
            
            if self.enable_reflection_rerun:
                should_continue_reflecting, feedback_info = self._perform_reflection()
                reflection_count = 0
                max_reflections = 1
                
                # Track best answer
                original_position = self._gt_position_before_reflection if hasattr(self, '_gt_position_before_reflection') else -1
                best_answer = self.answer.copy() if isinstance(self.answer, list) else self.answer
                best_position = original_position
                
                while should_continue_reflecting and reflection_count < max_reflections:
                    reflection_count += 1
                    logger.debug(f'Starting MARCO-P reflection cycle {reflection_count}/{max_reflections}')
                    
                    position_before = self._get_ground_truth_position(self._last_final_answer if hasattr(self, '_last_final_answer') else self.answer)
                    
                    self.reset(preserve_progress=True)
                    result = self._execute_workflow()
                    
                    position_after = self._get_ground_truth_position(self.answer)
                    
                    # Track rerun
                    sample_idx = self._current_sample_idx
                    user_id = self._current_user_id
                    gt_item = self.gt_answer if hasattr(self, 'gt_answer') else -1
                    
                    rerun_info = {
                        'sample_idx': sample_idx,
                        'user_id': user_id,
                        'gt_item': gt_item,
                        'position_before': position_before,
                        'position_after': position_after
                    }
                    self.reflection_all_reruns.append(rerun_info)
                    
                    # Track improvement
                    if position_before > 0 and position_after > 0 and position_after < position_before:
                        self.reflection_improvements.append(rerun_info)
                        best_answer = self.answer.copy() if isinstance(self.answer, list) else self.answer
                        best_position = position_after
                        logger.info(f"Reflection improved GT position: {position_before} → {position_after}")
                    
                    should_continue_reflecting, feedback_info = self._perform_reflection()
                
                # Restore best answer
                if best_position != self._get_ground_truth_position(self.answer):
                    self.answer = best_answer.copy() if isinstance(best_answer, list) else best_answer
            else:
                # Just log reflection without rerun
                self._perform_reflection_logging_only()
            
            return self.answer
                
        except Exception as e:
            logger.error(f"Error in MARCO-P forward: {e}")
            raise

    def _execute_workflow(self) -> str:
        logger.info("MARCO-P Workflow Phase 1: Analyst Analysis")
        
        # DEBUG: Check analyst max_turns BEFORE forward
        logger.info(f"DEBUG BEFORE forward: Analyst max_turns = {self.analyst.max_turns if self.analyst else 'NO ANALYST'}, _history length = {len(self.analyst._history) if self.analyst else 'NO ANALYST'}")
        
        candidate_items_str = self._extract_candidate_items()
        
        self.analyst.system = self
        
        # Call Analyst's forward method
        user_id = self._get_user_id()
        
        # Prepare analyst kwargs - only include what the analyst needs (no max_step in prompts)
        analyst_kwargs = {
            'task_type': self.manager_kwargs.get('task_type', self.task),
            'input': self.manager_kwargs.get('input', self.input),
            'data_sample': self.manager_kwargs.get('data_sample', self.data_sample),
            'candidate_items': candidate_items_str,
        }
        
        with duration_tracker.track_agent_call('analyst'):
            analyst_insights = self.analyst.forward(
                id=user_id, 
                analyse_type='user', 
                **analyst_kwargs
            )
        
        # DEBUG: Check analyst state AFTER forward
        logger.info(f"DEBUG AFTER forward: Analyst max_turns = {self.analyst.max_turns}, _history length = {len(self.analyst._history)}, finished = {self.analyst.finished}")
        
        self.log(f"**Analyst Insights:**\n{analyst_insights}", agent=self.analyst)
        
        # Store analyst insights
        self.execution_results['#E1'] = analyst_insights
        
        logger.info("MARCO-P Workflow Phase 2: Solver Synthesis")
        
        # Generate final solution using Solver
        with duration_tracker.track_agent_call('solver'):
            solution = self.solver.invoke_ablation(
                self.input, 
                analyst_insights, 
                self.task, 
                **self.manager_kwargs
            )
        
        self.log(f"**Solver Solution:**\n{solution}", agent=self.solver)
        
        final_answer = self.solver.extract_final_answer(solution, self.task)
        
        # Filter out history items for ranking tasks
        if self.task in ['sr'] and isinstance(final_answer, list):
            final_answer = self._filter_history_items(final_answer)
        
        # Store for reflection
        self._last_solution = solution
        self._last_final_answer = final_answer
        self._gt_position_before_reflection = self._get_ground_truth_position(final_answer)
        self._answer_before_reflection = final_answer.copy() if isinstance(final_answer, list) else final_answer
        
        logger.info(f"MARCO-P Final Answer: {final_answer} | Ground Truth: {self.gt_answer}")
        if self._gt_position_before_reflection > 0:
            logger.info(f"Ground truth position: {self._gt_position_before_reflection}")
        
        self.phase = 'completed'
        return self.finish(final_answer)
    
    def _filter_history_items(self, answer: list) -> list:
        history_item_ids = set()
        candidate_item_ids = set()
        
        if hasattr(self, 'data_sample') and self.data_sample is not None and 'history_item_id' in self.data_sample:
            try:
                history_item_id_value = self.data_sample['history_item_id']
                if isinstance(history_item_id_value, str):
                    history_item_ids = set(eval(history_item_id_value))
                elif isinstance(history_item_id_value, (list, set)):
                    history_item_ids = set(history_item_id_value)
            except Exception as e:
                logger.warning(f"Failed to extract history_item_id: {e}")
        
        if hasattr(self, 'data_sample') and self.data_sample is not None and 'candidate_item_id' in self.data_sample:
            try:
                candidate_item_id_value = self.data_sample['candidate_item_id']
                if isinstance(candidate_item_id_value, str):
                    candidate_item_ids = set(eval(candidate_item_id_value))
                elif isinstance(candidate_item_id_value, (list, set)):
                    candidate_item_ids = set(candidate_item_id_value)
            except Exception as e:
                logger.warning(f"Failed to extract candidate_item_id: {e}")
        
        # Filter answer
        original_answer = answer.copy()
        filtered_answer = [item_id for item_id in answer if item_id in candidate_item_ids]
        
        if len(filtered_answer) != len(original_answer):
            removed_items = [item_id for item_id in original_answer if item_id not in filtered_answer]
            logger.warning(f"Solver included history items! Removed: {removed_items}")
        
        return filtered_answer
    
    def _extract_candidate_items(self) -> str:
        if not hasattr(self, 'data_sample') or self.data_sample is None:
            return "No candidate items available"
        
        if 'candidate_item_id' not in self.data_sample:
            return "No candidate items available"
        
        try:
            candidate_item_id_value = self.data_sample['candidate_item_id']
            if isinstance(candidate_item_id_value, str):
                candidate_list = eval(candidate_item_id_value)
            elif isinstance(candidate_item_id_value, list):
                candidate_list = candidate_item_id_value
            else:
                return "No candidate items available"
            
            return f"Candidate item IDs: {candidate_list}"
        except Exception as e:
            logger.warning(f"Failed to extract candidate items: {e}")
            return "No candidate items available"
    
    def _get_user_id(self) -> int:
        if hasattr(self, 'data_sample') and self.data_sample is not None and 'user_id' in self.data_sample:
            return int(self.data_sample['user_id'])
        elif hasattr(self, '_current_user_id'):
            return self._current_user_id
        elif 'user_id' in self.kwargs:
            return int(self.kwargs['user_id'])
        else:
            return 1
    
    def _get_ground_truth_position(self, answer: Any) -> int:
        if not isinstance(answer, list) or not hasattr(self, 'gt_answer'):
            return -1
        
        gt = self.gt_answer
        try:
            if gt in answer:
                return answer.index(gt) + 1
        except (ValueError, TypeError):
            pass
        
        return -1
    
    def _perform_reflection(self) -> tuple[bool, dict]:
        if not self.reflector:
            return False, {}
        
        logger.info("🔍 Performing MARCO-P reflection")
        
        scratchpad = self._build_scratchpad()
        
        # Use reflector
        with duration_tracker.track_agent_call('reflector'):
            self.reflector(input=self.input, scratchpad=scratchpad)
        
        feedback_info = {
            'correct': True,
            'reason': ''
        }
        
        if self.reflector.json_mode and self.reflector.reflections:
            try:
                reflection_json = json.loads(self.reflector.reflections[-1])
                
                if isinstance(reflection_json, dict):
                    correctness = reflection_json.get('correctness', False)
                    reason = reflection_json.get('reason', 'No reason provided')
                    
                    feedback_info['correct'] = correctness
                    feedback_info['reason'] = reason
                    
                    if not correctness:
                        logger.debug(f"MARCO-P Reflection identified issues: {reason}")
                        self.log(f"**MARCO-P Reflection Issues:**\n{reason}", agent=self.reflector)
                        self.total_reflections_triggered += 1
                        return True, feedback_info
                    else:
                        logger.debug(f"MARCO-P Reflection confirms correctness: {reason}")
                        self.log(f"**MARCO-P Reflection Confirms:**\n{reason}", agent=self.reflector)
                        return False, feedback_info
                        
            except Exception as e:
                logger.error(f'Invalid reflection JSON: {e}')
                return False, feedback_info
        
        return False, feedback_info
    
    def _perform_reflection_logging_only(self) -> None:
        if not self.reflector:
            return
        
        scratchpad = self._build_scratchpad()
        
        with duration_tracker.track_agent_call('reflector'):
            self.reflector(input=self.input, scratchpad=scratchpad)
        
        if self.reflector.reflections:
            self.log(f"**MARCO-P Reflection:**\n{self.reflector.reflections[-1]}", agent=self.reflector)
    
    def _build_scratchpad(self) -> str:
        scratchpad = f"MARCO-P Process Summary\n"
        scratchpad += f"Task: {self.task.upper()}\n"
        scratchpad += f"Query: {getattr(self, 'input', 'No input')}\n\n"
        
        scratchpad += "Phase 1 - Analyst Analysis:\n"
        if self.execution_results.get('#E1'):
            scratchpad += f"{self.execution_results['#E1']}\n\n"
        else:
            scratchpad += "No analyst insights\n\n"
        
        scratchpad += "Phase 2 - Solving:\n"
        if hasattr(self, '_last_solution'):
            scratchpad += f"Solution: {self._last_solution}\n"
        if hasattr(self, '_last_final_answer'):
            scratchpad += f"Final Answer: {self._last_final_answer}\n"
        
        return scratchpad

    def is_finished(self) -> bool:
        return hasattr(self, 'finished') and self.finished

    def is_halted(self) -> bool:
        return self.phase == 'completed' and not self.is_finished()
