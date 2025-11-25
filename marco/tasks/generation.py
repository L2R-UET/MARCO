import os
import pandas as pd
from abc import abstractmethod
from tqdm import tqdm
from typing import Any
from loguru import logger
from argparse import ArgumentParser
import datetime

from marco.tasks.base import Task
from marco.utils import init_api, read_json, token_tracker, duration_tracker
from marco.utils.prompt_builder import PromptBuilder
from marco.systems import MARCOSystem, MARCOPSystem

class GenerationTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--dataset', type=str, default='None', help='Dataset name')
        parser.add_argument('--data_file', type=str, required=True, help='Dataset file')
        parser.add_argument('--system', type=str, default='marco', choices=['marco', 'marco_p'], help='System name')
        parser.add_argument('--system_config', type=str, required=True, help='System configuration file')
        parser.add_argument('--task', type=str, default='sr', choices=['rp', 'sr', 'gen'], help='Task name')
        parser.add_argument('--max_his', type=int, default=10, help='Max history length')
        
        parser.add_argument('--provider', type=str, choices=['openrouter', 'openai', 'ollama', 'gemini'], help='LLM provider type (e.g., openrouter, openai, ollama, gemini)')
        parser.add_argument('--model', type=str, help='Model name/version to use (e.g., google/gemini-2.0-flash-001, gpt-4o-mini, llama3.2:1b). If not specified, uses default for the provider.')
        parser.add_argument('--disable-reflection-rerun', action='store_false', dest='enable_reflection_rerun', help='Disable automatic rerun when reflector returns correctness: false (only for MARCO system)')

        return parser

    def get_data(self, data_file: str, max_his: int) -> pd.DataFrame:
        df = pd.read_csv(data_file)
        
        data_dir = os.path.dirname(data_file)
        self.prompt_builder = PromptBuilder(data_dir, self.dataset)
        
        if self.task == 'sr' and 'candidate_item_id' in df.columns:
            import ast
            first_candidates = df['candidate_item_id'].iloc[0]
            if isinstance(first_candidates, str):
                try:
                    first_candidates = ast.literal_eval(first_candidates)
                except:
                    pass
            
            if isinstance(first_candidates, list):
                self.n_candidate = len(first_candidates)
                self.system_kwargs['n_candidate'] = self.n_candidate
                logger.info(f"Detected {self.n_candidate} candidates for SR task")
        
        return df

    def prompt_data(self, df: pd.DataFrame) -> list[tuple[str, int | float | str, pd.Series]]:
        import ast
        
        if self.task == 'sr' and 'candidate_item_id' in df.columns:
            logger.info(f"Pre-filtering samples for {self.task} task...")
            valid_indices = []
            skipped_count = 0
            
            for i in range(len(df)):
                row = df.iloc[i]
                gt_item = row['item_id']
                candidate_ids = row['candidate_item_id']
                
                if isinstance(candidate_ids, str):
                    try:
                        candidate_ids = ast.literal_eval(candidate_ids)
                    except:
                        logger.warning(f"Failed to parse candidate_item_id for sample {i+1}, skipping")
                        skipped_count += 1
                        continue
                
                if isinstance(candidate_ids, list):
                    if gt_item in candidate_ids:
                        valid_indices.append(i)
                    else:
                        logger.trace(f"Skipping sample {i+1} (User {row['user_id']}): GT item {gt_item} not in candidates")
                        skipped_count += 1
                else:
                    valid_indices.append(i)
            
            if skipped_count > 0:
                logger.warning(f"Pre-filtered: Skipped {skipped_count}/{len(df)} samples where GT item not in candidates")
                df = df.iloc[valid_indices].reset_index(drop=True)
                logger.success(f"Building prompts for {len(df)} valid samples (filtered from {len(df) + skipped_count} total)")
            else:
                logger.info(f"All {len(df)} samples have GT in candidates")
        
        data_prompt = self.system.prompts['data_prompt']
        prompts = []
        
        logger.info(f"Building prompts for {len(df)} samples...")
        
        for i in tqdm(range(len(df)), desc="Building prompts"):
            row = df.iloc[i]
            
            fields = self.prompt_builder.build_prompt_fields(row, max_his=self.max_his)
            
            if self.task == 'rp':
                prompt = data_prompt.format(
                    user_id=row['user_id'],
                    user_profile=fields['user_profile'],
                    history=fields['history'],
                    target_item_id=row['item_id'],
                    target_item_attributes=fields['target_item_attributes']
                )
                target = row['rating']
            
            elif self.task == 'sr':
                prompt = data_prompt.format(
                    user_id=row['user_id'],
                    user_profile=fields['user_profile'],
                    history=fields['history'],
                    candidate_item_attributes=fields['candidate_item_attributes']
                )
                target = row['item_id']
            
            elif self.task == 'gen':
                prompt = data_prompt.format(
                    user_id=row['user_id'],
                    user_profile=fields['user_profile'],
                    history=fields['history'],
                    target_item_id=row['item_id'],
                    target_item_attributes=fields['target_item_attributes'],
                    rating=row['rating']
                )
                target = row['rating']
            
            else:
                raise NotImplementedError(f"Task {self.task} not implemented")
            
            prompts.append((prompt, target, row))
        
        logger.info(f"Built {len(prompts)} prompts")
        return prompts

    def get_system(self, system: str, system_config: str):
        if system == 'marco':
            self.system = MARCOSystem(config_path=system_config, **self.system_kwargs)
        elif system == 'marco_p':
            self.system = MARCOPSystem(config_path=system_config, **self.system_kwargs)
        else:
            raise NotImplementedError(f"Unknown system: {system}. Only 'marco' and 'marco_p' systems are available.")

    @property
    @abstractmethod
    def running_steps(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def before_generate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def after_step(self, answer: Any, gt_answer: int | float | str, step: int, record: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def after_iteration(self, answer: Any, gt_answer: int | float | str, record: dict, pbar: tqdm) -> None:
        raise NotImplementedError

    @abstractmethod
    def after_generate(self) -> None:
        raise NotImplementedError

    def generate(self, data: list[tuple[str, int | float | str, pd.Series]], steps: int = 2):
        task_id = f"{self.dataset}_{self.task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task_info = {
            'dataset': self.dataset,
            'task': self.task,
            'system': self.system.__class__.__name__,
            'model_override': self.model_override,
            'samples': len(data),
            'steps': steps,
            'max_history': self.max_his
        }
        
        token_tracker.start_task(task_id, task_info)
        duration_tracker.start_task(task_id, task_info)
        
        token_tracker.reset_agent_stats(self.system)
        
        self.before_generate()
        with tqdm(total=len(data)) as pbar:
            for sample_idx, (test_data, gt_answer, data_sample) in enumerate(data):
                sample_id = sample_idx + 1
                logger.info(f"Sample: {sample_id}/{len(data)}")
                
                record = dict()
                record['sample_id'] = sample_id
                record['user_id'] = data_sample.get('user_id', 'unknown')
                
                self.system.set_data(input=test_data, context="", gt_answer=gt_answer, data_sample=data_sample)
                self.system._current_sample_idx = sample_id
                self.system._current_user_id = record['user_id']
                self.system.reset(clear=True)
                
                for i in range(steps):
                    logger.debug(f'===================================Running step {i}...===================================')
                    self.after_step(answer=self.system(), gt_answer=gt_answer, step=i, record=record)
                    
                token_tracker.collect_system_stats(self.system)
                
                self.after_iteration(answer=self.system.answer, gt_answer=gt_answer, record=record, pbar=pbar)
                pbar.update(1)
                
        final_stats = token_tracker.end_task()
        duration_stats = duration_tracker.end_task()
        
        logger.success("=== Token Usage Summary ===")
        logger.success(f"Task: {self.dataset} {self.task} ({len(data)} samples)")
        logger.success(f"Total API calls: {final_stats.get('total_api_calls', 0)}")
        logger.success(f"Total tokens: {final_stats.get('total_tokens', 0)}")
        logger.success(f"Input tokens: {final_stats.get('total_input_tokens', 0)}")
        logger.success(f"Output tokens: {final_stats.get('total_output_tokens', 0)}")
        logger.success(f"Models used: {final_stats.get('models_used', [])}")
        logger.success(f"Duration: {final_stats.get('duration', 0):.2f}s")
        
        agents = final_stats.get('agents', {})
        agent_durations = duration_stats.get('agents', {})
        
        if agents or agent_durations:
            logger.success("=== Per-Agent Statistics===")
            all_agent_names = set(agents.keys()) | set(agent_durations.keys())
            
            for agent_name in sorted(all_agent_names):
                logger.success(f"Agent: {agent_name}")
                
                # Token usage info
                if agent_name in agents:
                    agent_stats = agents[agent_name]
                    logger.success(f"  API calls: {agent_stats.get('api_calls', 0)}")
                    logger.success(f"  Total tokens: {agent_stats.get('total_tokens', 0)}")
                    logger.success(f"  Input tokens: {agent_stats.get('total_input_tokens', 0)}")
                    logger.success(f"  Output tokens: {agent_stats.get('total_output_tokens', 0)}")
                    logger.success(f"  Model: {agent_stats.get('model', 'unknown')}")
                
                # Execution duration info
                if agent_name in agent_durations:
                    duration_info = agent_durations[agent_name]
                    logger.success(f"  Total duration: {duration_info.get('total_duration', 0):.3f}s")
                    logger.success(f"  Number of calls: {duration_info.get('call_count', 0)}")
                    logger.success(f"  Average duration per call: {duration_info.get('avg_duration_per_call', 0):.3f}s")
        
        self.after_generate()

    def run(self, api_config: str, dataset: str, data_file: str, system: str, system_config: str, task: str, max_his: int, provider: str = None, model: str = None, enable_reflection_rerun: bool = True):
        if dataset == 'None':
            dataset = os.path.basename(os.path.dirname(data_file))
        self.dataset = dataset
        self.task = task
        self.max_his = max_his
        
        data_dir = os.path.dirname(data_file)
        
        init_api(read_json(api_config))
        
        if provider:
            provider_info = self._parse_provider_options(provider, model)
            self.model_override = provider_info['model']
            self.provider = provider_info['provider']
            self.system_kwargs = {
                'task': self.task,
                'leak': False,
                'dataset': self.dataset,
                'data_dir': data_dir,
                'model_override': self.model_override,
                'provider': self.provider,
                'enable_reflection_rerun': enable_reflection_rerun,
            }
            logger.info(f"Using {provider_info['provider']} with model: {provider_info['model']} (will override all agents except opensource)")
        else:
            self.model_override = None
            self.provider = None
            self.system_kwargs = {
                'task': self.task,
                'leak': False,
                'dataset': self.dataset,
                'data_dir': data_dir,
                'enable_reflection_rerun': enable_reflection_rerun,
            }
            logger.info(f"No provider/model specified - using individual agent configurations")
        
        data_df = self.get_data(data_file, max_his)
        
        self.get_system(system, system_config)
        data = self.prompt_data(data_df)
        
        self.setup_task_logger(task=task, dataset=dataset, system=system, num_samples=len(data))
        
        self.generate(data, steps=self.running_steps)
    
    def _parse_provider_options(self, provider: str, model: str = None) -> dict:
        def _get_default(provider_name: str) -> str:
            try:
                from marco.utils import read_json
                api_config = read_json('config/api-config.json')
                if 'providers' in api_config and provider_name in api_config['providers']:
                    provider_cfg = api_config['providers'][provider_name]
                    for k in ('model', 'default_model'):
                        if provider_cfg.get(k):
                            return provider_cfg.get(k)
            except Exception:
                pass
            default_map = {
                'openrouter': 'google/gemini-2.0-flash-001',
                'openai': 'gpt-4o-mini',
                'ollama': 'llama3.2:1b',
                'gemini': 'google/gemini-2.0-flash-001'
            }
            return default_map.get(provider_name, 'google/gemini-2.0-flash-001')

        if not provider:
            return {
                'provider': None,
                'model': None
            }
        
        chosen_model = model if model else _get_default(provider)
        
        if provider == 'openai':
            chosen_model = self._normalize_openai_model(chosen_model)
        
        return {
            'provider': provider,
            'model': chosen_model
        }

    @staticmethod
    def _normalize_openai_model(model: str) -> str:
        if not model:
            return 'gpt-4o-mini'
        cleaned = model.strip()
        if '/' in cleaned:
            prefix, suffix = cleaned.split('/', 1)
            if prefix.lower() == 'openai':
                return suffix
        return cleaned
