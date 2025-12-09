import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from loguru import logger

from marco.llms.basellm import BaseLLM
from marco.utils import read_json

# Default constants (fallback values)
DEFAULT_CONFIG_PATH = 'config/api-config.json'
DEFAULT_MODEL = 'gpt-4o-mini'
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_BASE = 1
DEFAULT_MAX_TOKENS_LIMIT = 4096
DEFAULT_CONTEXT_LENGTH = 128000
DEFAULT_AGENT_CONTEXT = "Unknown"

class OpenAILLM(BaseLLM):

    def __init__(
        self,
        model: str = '',
        api_key: str = '',
        json_mode: bool = False,
        agent_context: Optional[str] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        # Load config to get defaults
        api_config = self._load_api_config()
        provider_cfg = api_config.get('providers', {}).get('openai', {}) if api_config else {}
        
        # Resolve model: explicit > config > default
        if not model:
            model = provider_cfg.get('model', '') or DEFAULT_MODEL
        
        self.original_model = model or DEFAULT_MODEL
        
        # Store provider_cfg for later use (detecting proxy, etc.)
        self._provider_cfg = provider_cfg
        
        self.json_mode = json_mode
        self.agent_context = agent_context or DEFAULT_AGENT_CONTEXT
        
        # Resolve parameters: kwargs > config > defaults
        self.max_tokens: int = kwargs.get('max_tokens') or provider_cfg.get('max_tokens') or DEFAULT_MAX_TOKENS
        self.temperature: float = kwargs.get('temperature') or provider_cfg.get('temperature') or DEFAULT_TEMPERATURE
        self.top_p: float = kwargs.get('top_p') or provider_cfg.get('top_p') or DEFAULT_TOP_P
        self.max_retries = kwargs.get('max_retries') or provider_cfg.get('max_retries') or DEFAULT_MAX_RETRIES
        self.retry_delay_base = kwargs.get('retry_delay_base') or provider_cfg.get('retry_delay_base') or DEFAULT_RETRY_DELAY_BASE

        # Resolve API key from explicit config, env vars, or api-config.json
        self.api_key = self._resolve_api_key(api_key)

        # Base URL for OpenAI - must be present in config, no hardcode fallback
        # Auto-detect if using official OpenAI or proxy provider
        try:
            if not provider_cfg or not provider_cfg.get('base_url'):
                raise ValueError(f"Missing 'base_url' for OpenAI provider in {DEFAULT_CONFIG_PATH}!")
            
            # Use base_url directly from config without normalization
            # Only remove trailing slash to avoid double slashes
            raw_base_url = provider_cfg['base_url'].strip()
            self.base_url = raw_base_url.rstrip('/')
            
            # Detect if this is official OpenAI or a proxy provider
            self.is_official_openai = self._is_official_openai_url(self.base_url)
            
            # Normalize model name based on provider type
            if self.is_official_openai:
                # Official OpenAI: normalize model name (remove prefix like "openai/")
                self.model = self._normalize_model_for_official(self.original_model)
                logger.info("Detected official OpenAI API endpoint")
            else:
                # Proxy provider: use model name as-is (support multi-provider models)
                # Proxy providers like megallm.io support models from different providers:
                # - gpt-5, gpt-4o-mini (OpenAI-compatible)
                # - xai/grok-code-fast-1 (xAI)
                # - claude-opus-4-1-20250805 (Anthropic)
                self.model = self._normalize_model_for_proxy(self.original_model, provider_cfg)
                logger.info(f"Detected proxy provider: {self.base_url}")
                logger.info(f"Using model: {self.model} (proxy supports multi-provider models)")
            
            self.max_context_length = self._get_context_length(self.model)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenAI base_url from config: {e}")

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAI LLMs. Install it with: pip install openai>=1.0.0"
            ) from exc

        # Instantiate the OpenAI client with base_url (proxy enforced)
        self.client = OpenAI(api_key=self.api_key or None, base_url=self.base_url)

        provider_type = "Official OpenAI" if self.is_official_openai else "Proxy Provider"
        logger.info(
            f"Initialized OpenAI LLM ({provider_type}) with model '{self.model}' "
            f"(display: '{self.original_model}'), agent='{self.agent_context}', "
            f"max_retries={self.max_retries}, base_url={self.base_url}"
        )

    @staticmethod
    def _load_api_config() -> Dict[str, Any]:
        try:
            return read_json(DEFAULT_CONFIG_PATH)
        except Exception as e:
            logger.warning(f"Failed to load API config from {DEFAULT_CONFIG_PATH}: {e}")
            return {}

    @staticmethod
    def _normalize_model_for_official(model: str) -> str:
        """Normalize model name for official OpenAI API.
        
        Strips optional provider prefix (e.g., openai/gpt-4o-mini -> gpt-4o-mini).
        """
        cleaned = (model or '').strip()
        if '/' in cleaned:
            prefix, suffix = cleaned.split('/', 1)
            if prefix.lower() == 'openai':
                return suffix
        return cleaned or DEFAULT_MODEL

    @staticmethod
    def _normalize_model_for_proxy(model: str, provider_cfg: Dict[str, Any]) -> str:
        """Normalize model name for proxy provider.
        
        Proxy providers support models from multiple AI providers:
        - OpenAI: gpt-5, gpt-4o-mini, etc.
        - xAI: xai/grok-code-fast-1, etc.
        - Anthropic: claude-opus-4-1-20250805, etc.
        
        Use model name as-is to support multi-provider models.
        Check if there's a model mapping in config for aliases.
        """
        cleaned = (model or '').strip()
        
        # Check for model aliases/mappings in config
        if provider_cfg and 'model_mappings' in provider_cfg:
            model_mappings = provider_cfg['model_mappings']
            if cleaned in model_mappings:
                mapped_model = model_mappings[cleaned]
                logger.info(f"Model alias '{cleaned}' mapped to '{mapped_model}'")
                return mapped_model
        
        # For proxy providers, preserve the full model name (including provider prefix if any)
        return cleaned or DEFAULT_MODEL

    @staticmethod
    def _is_official_openai_url(url: str) -> bool:
        """Detect if the URL is the official OpenAI API endpoint.
        
        Args:
            url: The base URL to check
            
        Returns:
            True if this is the official OpenAI API, False if it's a proxy
        """
        if not url:
            return False
        
        url_lower = url.lower().strip()
        
        # Official OpenAI domains (exact match in hostname, not just substring)
        # This prevents false positives like "my-openai-proxy.com"
        official_domains = [
            'api.openai.com',
            'openai.com',
            'https://api.openai.com',
            'http://api.openai.com',
        ]
        
        # Check for exact domain match in the URL
        try:
            parsed = urlparse(url_lower)
            hostname = parsed.netloc or parsed.path.split('/')[0] if not parsed.netloc else parsed.netloc
            
            if ':' in hostname:
                hostname = hostname.split(':')[0]
            
            for domain in official_domains:
                domain_clean = domain.replace('https://', '').replace('http://', '')
                if hostname == domain_clean or hostname.endswith('.' + domain_clean):
                    return True
        except Exception:
            # Fallback: simple substring check (less accurate but safer)
            for domain in ['api.openai.com', 'openai.com']:
                if domain in url_lower:
                    return True
        
        return False

    @staticmethod
    def _get_context_length(model: str) -> int:
        context_map = {
            'gpt-4o': DEFAULT_CONTEXT_LENGTH,
            'gpt-4o-mini': DEFAULT_CONTEXT_LENGTH,
            'gpt-4.1-mini': DEFAULT_CONTEXT_LENGTH,
            'gpt-4.1': DEFAULT_CONTEXT_LENGTH,
            'o4-mini': 200000,
            'o4-mini-high': 200000,
            'gpt-4-turbo': DEFAULT_CONTEXT_LENGTH,
            'gpt-4': 8192,
            'gpt-3.5-turbo': 16384,
        }
        return context_map.get(model, DEFAULT_CONTEXT_LENGTH)

    def _resolve_api_key(self, explicit_key: str) -> str:
        if explicit_key:
            return explicit_key
        env_key = os.getenv('OPENAI_API_KEY')
        if env_key:
            return env_key
        try:
            api_config = self._load_api_config()
            provider_cfg = api_config.get('providers', {}).get('openai', {})
            key = provider_cfg.get('api_key', '')
            if key:
                return key
            # Fallback to legacy structure
            provider = api_config.get('provider', '').lower()
            if provider == 'openai':
                return api_config.get('api_key', '') or api_config.get('openai_api_key', '')
            if 'openai_api_key' in api_config:
                return api_config['openai_api_key']
        except Exception:
            pass
        logger.warning(f"OpenAI API key not found. Set OPENAI_API_KEY env var or update {DEFAULT_CONFIG_PATH}.")
        return ''

    def _get_retriable_errors(self) -> tuple:
        base_errors = super()._get_retriable_errors()
        try:
            from openai import APIConnectionError, APITimeoutError, RateLimitError  # type: ignore
            return base_errors + (APIConnectionError, APITimeoutError, RateLimitError)
        except ImportError:
            return base_errors

    def _is_retriable_exception(self, exception: Exception) -> bool:
        try:
            from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError  # type: ignore

            if isinstance(exception, (APIConnectionError, APITimeoutError, RateLimitError)):
                return True
            if isinstance(exception, APIError):
                status = getattr(exception, 'status', None)
                return status == 429 or (isinstance(status, int) and status >= 500)
        except ImportError:
            pass

        return super()._is_retriable_exception(exception)

    def _make_api_request(
        self,
        messages: list[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
    ) -> Any:
        logger.info(f"[API CALL] Provider: openai | Model: {self.model}")
        request_payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": min(self.max_tokens, DEFAULT_MAX_TOKENS_LIMIT),
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        if response_format:
            request_payload["response_format"] = response_format

        return self.client.chat.completions.create(**request_payload)

    @staticmethod
    def _extract_message_content(response: Any) -> str:
        choices = getattr(response, 'choices', None)
        if not choices:
            return ""

        message = getattr(choices[0], 'message', None)
        if not message:
            return ""

        content = getattr(message, 'content', None)
        if isinstance(content, str):
            return content.strip()

        # Responses API may return list of content parts
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get('type') == 'text':
                        parts.append(part.get('text', ''))
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts).strip()

        return str(content or "").strip()

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        try:
            # Log the prompt being sent to the API (skip for Analyst to reduce log spam)
            if self.agent_context != "Analyst":
                logger.info(f"LLM Prompt ({self.agent_context} | {self.model}):\n{prompt}")
            estimated_prompt_tokens = self.estimate_tokens(prompt)
            logger.info(
                f"Token usage estimate ({self.agent_context}): ~{estimated_prompt_tokens} prompt tokens"
            )

            if isinstance(prompt, list):
                # If prompt is already a list of messages (dicts), use it directly
                messages = prompt
                # If using json mode and the last message is user, append instruction
                if self.json_mode and messages and messages[-1]['role'] == 'user':
                    # Load config to get json_instruction if available
                    api_config = self._load_api_config()
                    provider_cfg = api_config.get('providers', {}).get('openai', {}) if api_config else {}
                    json_instruction = provider_cfg.get('json_instruction', 'Return valid JSON only.')
                    
                    # Clone to avoid mutating original list if needed, or just append to content
                    original_content = messages[-1].get('content', '')
                    messages[-1]['content'] = f"{original_content}\n\n{json_instruction}"
            else:
                # Prompt is a string
                messages = [{"role": "user", "content": prompt}]
                if self.json_mode:
                    # Load config to get json_instruction if available
                    api_config = self._load_api_config()
                    provider_cfg = api_config.get('providers', {}).get('openai', {}) if api_config else {}
                    json_instruction = provider_cfg.get('json_instruction', 'Return valid JSON only.')
                    messages[0]["content"] = f"{prompt}\n\n{json_instruction}"

            response_format = {"type": "json_object"} if self.json_mode else None

            response = self.execute_with_retry(
                self._make_api_request,
                messages=messages,
                response_format=response_format,
            )

            content = self._extract_message_content(response)
            usage_info = getattr(response, 'usage', None)
            input_tokens = getattr(usage_info, 'prompt_tokens', None) if usage_info else None
            output_tokens = getattr(usage_info, 'completion_tokens', None) if usage_info else None

            usage_dict: Dict[str, Any] = {}
            if usage_info:
                usage_dict = {
                    "prompt_tokens": getattr(usage_info, 'prompt_tokens', None),
                    "completion_tokens": getattr(usage_info, 'completion_tokens', None),
                    "total_tokens": getattr(usage_info, 'total_tokens', None),
                }

            self.track_usage(
                prompt,
                content,
                input_tokens,
                output_tokens,
                api_usage=usage_dict,
            )

            logger.info(f"LLM Response ({self.agent_context} | {self.model}):\n{content}")

            if usage_info:
                prompt_tokens = usage_dict.get('prompt_tokens') or 0
                completion_tokens = usage_dict.get('completion_tokens') or 0
                total_tokens = usage_dict.get('total_tokens') or (prompt_tokens + completion_tokens)
                logger.info(
                    f"Token usage ({self.agent_context}): "
                    f"{prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total"
                )

            return content.strip()

        except Exception as exc:
            return self.handle_api_error(exc)
