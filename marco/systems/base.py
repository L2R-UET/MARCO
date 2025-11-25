import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Optional
from loguru import logger

from marco.agents import Agent
from marco.utils import is_correct, init_answer, read_json, read_prompts, get_avatar

class System(ABC):
    @staticmethod
    @abstractmethod
    def supported_tasks() -> list[str]:
        raise NotImplementedError("System.supported_tasks() not implemented")

    @property
    def task_type(self) -> str:
        if self.task == 'qa':
            return 'question answering'
        elif self.task == 'rp':
            return 'rating prediction'
        elif self.task == 'sr':
            return 'ranking'
        elif self.task == 'chat':
            return 'conversation'
        elif self.task == 'gen':
            return 'explanation generation'
        else:
            # Fallback: use raw task name to avoid crashing in UI flows for new tasks
            return self.task

    def __init__(self, task: str, config_path: str, leak: bool = False, dataset: Optional[str] = None, *args, **kwargs) -> None:
        self.task = task
        assert self.task in self.supported_tasks()
        self.config = read_json(config_path)
        if 'supported_tasks' in self.config:
            assert isinstance(self.config['supported_tasks'], list) and self.task in self.config['supported_tasks'], f'Task {self.task} is not supported by the system.'
        
        # No default provider - only override if explicitly provided
        self.model_override = kwargs.get('model_override', None)
        self.provider = kwargs.get('provider', None)
        
        self.agent_kwargs = {
            'system': self,
        }
        if dataset is not None:
            for key, value in self.config.items():
                if isinstance(value, str):
                    self.config[key] = value.format(dataset=dataset, task=self.task)
            self.agent_kwargs['dataset'] = dataset
        
        if 'data_dir' in kwargs:
            self.agent_kwargs['data_dir'] = kwargs['data_dir']
        
        self.prompts = {}
        self.prompts.update(read_prompts(self.config['data_prompt'].format(task=self.task)))
        self.agent_kwargs['prompts'] = self.prompts
        self.leak = leak
        self.kwargs = kwargs
        self.init(*args, **kwargs)
        self.reset(clear=True)

    def _apply_model_override(self, config: dict) -> dict:
        """Apply model override to a configuration dict.

        Only overrides if both model_override and provider are explicitly provided.
        Otherwise, uses the provider and model from the agent's config file.
        Skips override for agents with provider type 'opensource'.
        """
        # Only override if BOTH model_override and provider are provided
        if not self.model_override or not self.provider:
            return config

        original_provider = config.get('provider', config.get('model_type', '')).lower()  # Support both 'provider' and legacy 'model_type'
        if original_provider == 'opensource':
            logger.debug(f"Skipping model override for opensource agent (keeping original config)")
            return config

        config = config.copy()

        config['provider'] = self.provider
        if 'model_type' in config:
            del config['model_type']
        config['model'] = self.model_override
        
        if self.provider == 'openrouter':
            try:
                api_config = read_json('config/api-config.json')
                openrouter_key = None
                if 'providers' in api_config and 'openrouter' in api_config['providers']:
                    openrouter_key = api_config['providers']['openrouter'].get('api_key')
                if not openrouter_key:
                    provider = api_config.get('provider', '').lower()
                    if provider == 'openrouter':
                        openrouter_key = api_config.get('api_key')
                    if not openrouter_key:
                        openrouter_key = api_config.get('openrouter_api_key')
                if openrouter_key:
                    config['api_key'] = openrouter_key
                    logger.info(f"Using OpenRouter API for model: {self.model_override}")
                else:
                    logger.warning("OpenRouter API key not found in config")
            except Exception as e:
                logger.warning(f"Could not read API config for OpenRouter model override: {e}")
        elif self.provider == 'openai':
            try:
                api_config = read_json('config/api-config.json')
                openai_key = None
                if 'providers' in api_config and 'openai' in api_config['providers']:
                    openai_key = api_config['providers']['openai'].get('api_key')
                if not openai_key:
                    provider = api_config.get('provider', '').lower()
                    if provider == 'openai':
                        openai_key = api_config.get('api_key') or api_config.get('openai_api_key')
                    if not openai_key:
                        openai_key = api_config.get('openai_api_key')

                if openai_key:
                    config['api_key'] = openai_key
                    logger.info(f"Using OpenAI API for model: {self.model_override}")
                else:
                    logger.warning(
                        "OpenAI API key not found in config; relying on OPENAI_API_KEY environment variable."
                    )
            except Exception as e:
                logger.warning(f"Could not read API config for OpenAI model override: {e}")
        elif self.provider == 'ollama':
            logger.info(f"Using Ollama local model: {self.model_override}")
        elif self.provider == 'gemini':
            try:
                api_config = read_json('config/api-config.json')
                gemini_key = None
                if 'providers' in api_config and 'gemini' in api_config['providers']:
                    gemini_key = api_config['providers']['gemini'].get('api_key')
                if not gemini_key:
                    provider = api_config.get('provider', '').lower()
                    if provider == 'gemini':
                        gemini_key = api_config.get('api_key') or api_config.get('gemini_api_key')
                    if not gemini_key:
                        gemini_key = api_config.get('gemini_api_key')

                if gemini_key:
                    config['api_key'] = gemini_key
                    logger.info(f"Using Gemini API for model: {self.model_override}")
                else:
                    logger.warning("Gemini API key not found in config")
            except Exception as e:
                logger.warning(f"Could not read API config for Gemini model override: {e}")
        else:
            logger.warning(f"Unknown provider: {self.provider}")
        
        return config

    def log(self, message: str, agent: Optional[Agent] = None, logging: bool = True) -> None:
        logger.debug(message)

    @abstractmethod
    def init(self, *args, **kwargs) -> None:
        raise NotImplementedError("System.init() not implemented")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def set_data(self, input: str, context: str, gt_answer: Any, data_sample: Optional[pd.Series] = None) -> None:
        self.input: str = input
        self.context: str = context
        self.gt_answer = gt_answer
        self.data_sample = data_sample

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("System.forward() not implemented")

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return is_correct(task=self.task, answer=self.answer, gt_answer=self.gt_answer)

    def finish(self, answer: Any) -> str:
        self.answer = answer
        if not self.leak:
            observation = f'The answer you give (may be INCORRECT): {self.answer}'
        elif self.is_correct():
            observation = 'Answer is CORRECT'
        else:
            observation = 'Answer is INCORRECT'
        self.finished = True
        return observation

    def reset(self, clear: bool = False, *args, **kwargs) -> None:
        self.scratchpad: str = ''
        self.finished: bool = False
        self.answer = init_answer(type=self.task)
