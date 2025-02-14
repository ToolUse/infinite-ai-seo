"""
ai.py

This module provides a centralized interface for making AI calls using litellm.
It handles structured responses using Pydantic models and provides consistent
error handling and retries for AI operations.

The module provides:
- Structured AI responses with Pydantic validation
- Consistent error handling and logging
- Model configuration management
- Retry logic for failed calls
"""

import os
from typing import TypeVar, Type, List, Dict, Any
from pydantic import BaseModel
from litellm import completion
import litellm
from src.utils.logger import log
import time

# Type variable for Pydantic model
T = TypeVar('T', bound=BaseModel)

class AIError(Exception):
    """Custom exception for AI-related errors"""
    pass

def structured_ai_response(
    messages: List[Dict[str, str]],
    response_model: Type[T],
    model: str,
    api_key: str,
    token_tracker=None,
    max_retries: int = 3,
    timeout: int = 90,
) -> T:
    """
    Make an AI call and return a structured response using a Pydantic model.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        response_model: Pydantic model class for response validation
        model: Model identifier (e.g., "openai/gpt-4", "anthropic/claude-2")
        api_key: API key to use for this request
        token_tracker: Token tracker object
        max_retries: Number of retry attempts on failure
        timeout: Request timeout in seconds
        
    Returns:
        Instance of the provided Pydantic model
        
    Raises:
        AIError: If the AI call fails or response validation fails
    """
    retries = 0
    last_error = None
    
    log.info(f"\n🤖 Making call to {model}")
    log.debug(f"  Config: max_retries={max_retries}, timeout={timeout}s")
    
    while retries <= max_retries:
        try:
            # Check Gemini request limits before making call
            if token_tracker and model.startswith("gemini/"):
                if token_tracker.check_request_limit(model):
                    # If we hit minute limit, wait and retry
                    if retries < max_retries:
                        wait_time = 75  # Wait 75 seconds to be safe
                        log.info(f"😴 Rate limited, waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        retries += 1
                        continue
                    else:
                        # If we've retried max times, try switching models
                        next_model = token_tracker.get_next_available_model(model)
                        if next_model:
                            log.info(f"🔄 Switching to {next_model} due to rate limits")
                            return structured_ai_response(
                                messages=messages,
                                response_model=response_model,
                                model=next_model,
                                api_key=api_key,
                                token_tracker=token_tracker,
                                max_retries=max_retries,
                                timeout=timeout
                            )
                        else:
                            raise AIError("Rate limit exceeded and no alternative models available")

            if retries > 0:
                log.info(f"🔄 Retry attempt {retries}/{max_retries}")
            
            log.debug("\n📡 Sending request to AI model...")
            completion_args = {
                "model": model,
                "messages": messages,
                "response_format": response_model,
                "timeout": timeout,
                "api_key": api_key
            }
            
            if model.startswith("ollama"):
                completion_args["api_base"] = "http://localhost:11434"
                
            response = completion(**completion_args)
            
            # Track the request for Gemini
            if token_tracker and model.startswith("gemini/"):
                token_tracker.track_request(model)

            log.success("\n✨ Successfully received AI response")
            
            # Track token usage if available and tracker exists
            if hasattr(response, 'usage') and token_tracker is not None:
                usage = response.usage
                
                # Handle different token counting for Gemini
                if model.startswith("gemini/"):
                    # Gemini counts total tokens differently
                    total_tokens = usage.total_tokens
                    # Approximate split for Gemini (adjust ratio as needed)
                    prompt_tokens = int(total_tokens * 0.6)
                    completion_tokens = total_tokens - prompt_tokens
                else:
                    # OpenAI style token counting
                    prompt_tokens = usage.prompt_tokens
                    completion_tokens = usage.completion_tokens
                    total_tokens = prompt_tokens + completion_tokens

                # Check if we would exceed limits
                if token_tracker.check_limit(model, total_tokens):
                    # Try to get next available model
                    next_model = token_tracker.get_next_available_model(model)
                    if next_model:
                        log.info(f"🔄 Switching to {next_model} due to token limits")
                        # Recursive call with new model
                        return structured_ai_response(
                            messages=messages,
                            response_model=response_model,
                            model=next_model,
                            api_key=api_key,
                            token_tracker=token_tracker,
                            max_retries=max_retries,
                            timeout=timeout
                        )
                    else:
                        raise AIError(f"Token limit would be exceeded for model {model} and no alternatives available")

                # Track the usage
                token_tracker.track_usage(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            
            # Parse and validate response
            try:
                content = str(response.choices[0].message.content if hasattr(response, 'choices') else response.message.content)
                if not content:
                    raise AIError("Empty response from AI model")
                
                # Log response preview
                preview = content[:200] + "..." if len(content) > 200 else content
                log.info(f"\n📥 Response Preview:\n{preview}")
                    
                # Validate and return
                validated_response = response_model.model_validate_json(content)
                log.debug(f"\n✅ Response validated against schema: {response_model.__name__}")
                return validated_response
                
            except Exception as ve:
                raise AIError(f"Failed to validate response against schema: {str(ve)}")
            
        except Exception as e:
            last_error = e
            
            # Handle Gemini rate limit errors
            if (isinstance(e, litellm.APIError) and 
                str(e).lower().find("resource_exhausted") >= 0 and
                model.startswith("gemini/")):
                
                if retries < max_retries:
                    wait_time = 75  # Wait 75 seconds to be safe
                    log.info(f"😴 Rate limited, waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    retries += 1
                    continue
                else:
                    # Try switching models after max retries
                    next_model = token_tracker.get_next_available_model(model)
                    if next_model:
                        log.info(f"🔄 Switching to {next_model} due to rate limits")
                        return structured_ai_response(
                            messages=messages,
                            response_model=response_model,
                            model=next_model,
                            api_key=api_key,
                            token_tracker=token_tracker,
                            max_retries=max_retries,
                            timeout=timeout
                        )
            
            # Handle other errors
            retries += 1
            log.warning(f"⚠️  AI call attempt {retries} failed: {str(e)}")
            
            if retries <= max_retries:
                continue
            
            log.error(f"❌ All AI call attempts failed for model {model}")
            raise AIError(f"AI call failed after {max_retries} retries: {last_error}")
    
    raise AIError("Unexpected end of function")

def format_system_prompt(prompt_template: str, **kwargs: Any) -> Dict[str, str]:
    """
    Format a system prompt template with provided variables.
    
    Args:
        prompt_template: The prompt template string
        kwargs: Variables to format into the template
        
    Returns:
        Formatted message dictionary
    """
    log.debug("🎯 Formatting system prompt")
    return {
        "role": "system",
        "content": prompt_template.format(**kwargs)
    }

def format_user_prompt(prompt_template: str, **kwargs: Any) -> Dict[str, str]:
    """
    Format a user prompt template with provided variables.
    
    Args:
        prompt_template: The prompt template string
        kwargs: Variables to format into the template
        
    Returns:
        Formatted message dictionary
    """
    log.debug("👤 Formatting user prompt")
    return {
        "role": "user",
        "content": prompt_template.format(**kwargs)
    }
