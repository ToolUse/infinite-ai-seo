"""
token_tracker.py

This module provides functionality to track token usage across different models,
maintaining both daily and lifetime statistics. It ensures we stay within our
daily token limits and provides usage insights.
"""

import json
from datetime import datetime
import pytz
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from contextlib import contextmanager

from src.utils.logger import log

class TokenTracker:
    def __init__(self, config):
        self.tracking_file = Path(config["token_tracking"]["tracking_file"])
        self.model_groups = config["token_tracking"]["model_groups"]
        self.timezone = pytz.timezone(config["token_tracking"]["timezone"])
        self.stop_at_daily_token_limit = config["token_tracking"]["stop_at_daily_token_limit"]
        self.auto_switch = config["token_tracking"].get("auto_switch", False)
        self.provider_priority = config["token_tracking"].get("provider_priority", ["openai", "gemini"])
        self._ensure_tracking_file()
        
        # Initialize request tracking for Gemini
        self._ensure_request_tracking()
        
    def _get_current_date(self) -> str:
        """Get current date in configured timezone."""
        return datetime.now(self.timezone).strftime("%Y-%m-%d")
        
    def _get_model_group(self, model: str) -> Tuple[str, int]:
        """
        Get the group name and limit for a given model.
        
        Returns:
            Tuple of (group_name, daily_limit)
        Raises:
            ValueError if model not found in any group
        """
        for group_name, group_info in self.model_groups.items():
            if model in group_info["models"]:
                return group_name, group_info["daily_limit"]
        raise ValueError(f"Model {model} not found in any configured group")
        
    def _ensure_tracking_file(self) -> None:
        """Ensure the token tracking file exists with valid initial data."""
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.tracking_file.exists():
            initial_data = {
                "last_update": self._get_current_date(),
                "daily": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                },
                "lifetime": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                },
                "model_groups": {
                    group_name: {
                        "daily_tokens": 0,
                        "lifetime_tokens": 0
                    }
                    for group_name in self.model_groups.keys()
                },
                "models": {},  # Keep individual model tracking for analytics
            }
            self._save_data(initial_data)
    
    def _load_data(self) -> Dict:
        """Load the current tracking data."""
        with open(self.tracking_file, 'r') as f:
            return json.load(f)
    
    def _save_data(self, data: Dict) -> None:
        """Save the tracking data to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _reset_daily_counts(self, data: Dict) -> Dict:
        """Reset daily token counts while preserving lifetime counts."""
        data["last_update"] = self._get_current_date()
        data["daily"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        # Reset daily counts for model groups
        for group in data["model_groups"]:
            data["model_groups"][group]["daily_tokens"] = 0
        # Reset daily counts for individual models
        for model in data["models"]:
            data["models"][model]["daily_tokens"] = 0
        return data
    
    def check_limit(self, model: str, tokens: int) -> bool:
        """
        Check if adding these tokens would exceed the daily limit for the model's group.
        
        Args:
            model: The model being used
            tokens: Number of tokens to be added
            
        Returns:
            bool: True if adding these tokens would exceed the limit
        """
        try:
            group_name, limit = self._get_model_group(model)
        except ValueError as e:
            log.warning(f"No limit configured for model {model} - allowing usage")
            return False
            
        data = self._load_data()
        current = data["model_groups"][group_name]["daily_tokens"]
        
        would_exceed = (current + tokens) > limit
        if would_exceed:
            log.error(f"⚠️  Adding {tokens} tokens would exceed daily limit of {limit} for group {group_name}")
            log.error(f"Current group usage: {current} tokens")
            log.error(f"This affects models: {', '.join(self.model_groups[group_name]['models'])}")
        return would_exceed
    
    def _get_provider_from_model(self, model: str) -> str:
        """Extract provider from model string"""
        if model.startswith("gemini/"):
            return "gemini"
        elif model.startswith("openai/"):
            return "openai"
        else:
            return "unknown"

    def get_next_available_model(self, current_model: str) -> Optional[str]:
        """
        Get the next available model when current model's tokens are exhausted.
        Returns None if no models are available.
        """
        if not self.auto_switch:
            return None
            
        current_provider = self._get_provider_from_model(current_model)
        data = self._load_data()
        
        # Try providers in priority order
        for provider in self.provider_priority:
            # Skip until we get to next provider after current
            if provider == current_provider:
                continue
                
            # Find models for this provider
            available_models = []
            for group_name, group_info in self.model_groups.items():
                for model in group_info["models"]:
                    if model.startswith(f"{provider}/"):
                        # Check if model has tokens available
                        group_usage = data["model_groups"][group_name]["daily_tokens"]
                        if group_usage < group_info["daily_limit"]:
                            available_models.append(model)
                            
            if available_models:
                return available_models[0]  # Return first available model
                
        return None

    def track_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Track token usage for an AI call."""
        data = self._load_data()
        
        # Check if we need to reset daily counts
        current_date = self._get_current_date()
        if data["last_update"] != current_date:
            data = self._reset_daily_counts(data)
        
        total_tokens = prompt_tokens + completion_tokens
        provider = self._get_provider_from_model(model)
        
        # Update daily counts
        data["daily"]["prompt_tokens"] += prompt_tokens
        data["daily"]["completion_tokens"] += completion_tokens
        data["daily"]["total_tokens"] += total_tokens
        
        # Update lifetime counts
        data["lifetime"]["prompt_tokens"] += prompt_tokens
        data["lifetime"]["completion_tokens"] += completion_tokens
        data["lifetime"]["total_tokens"] += total_tokens
        
        # Initialize provider tracking if needed
        if "providers" not in data:
            data["providers"] = {}
        if provider not in data["providers"]:
            data["providers"][provider] = {
                "daily_tokens": 0,
                "lifetime_tokens": 0
            }
        
        # Update provider-specific counts
        data["providers"][provider]["daily_tokens"] += total_tokens
        data["providers"][provider]["lifetime_tokens"] += total_tokens
        
        # Update model-specific counts
        if model not in data["models"]:
            data["models"][model] = {
                "daily_tokens": 0,
                "lifetime_tokens": 0
            }
        data["models"][model]["daily_tokens"] += total_tokens
        data["models"][model]["lifetime_tokens"] += total_tokens
        
        # Update model group counts
        try:
            group_name, _ = self._get_model_group(model)
            if group_name not in data["model_groups"]:
                data["model_groups"][group_name] = {
                    "daily_tokens": 0,
                    "lifetime_tokens": 0
                }
            data["model_groups"][group_name]["daily_tokens"] += total_tokens
            data["model_groups"][group_name]["lifetime_tokens"] += total_tokens
        except ValueError:
            log.warning(f"Model {model} not found in any group - skipping group tracking")
        
        
        self._save_data(data)
        
        # Enhanced logging
        log.info(f"\n📊 Token Usage ({provider.upper()}):")
        log.info(f"  Model: {model}")
        log.info(f"  Prompt tokens: {prompt_tokens}")
        log.info(f"  Completion tokens: {completion_tokens}")
        log.info(f"  Total tokens: {total_tokens}")
        
        # Log group usage and check for switching
        try:
            group_name, limit = self._get_model_group(model)
            group_usage = data["model_groups"][group_name]["daily_tokens"]
            usage_percent = (group_usage / limit) * 100
            log.info(f"  Group: {group_name}")
            log.info(f"  Group usage: {group_usage}/{limit} tokens ({usage_percent:.1f}%)")
            
            if usage_percent > 80:
                next_model = self.get_next_available_model(model)
                if next_model:
                    log.warning(f"⚠️  Consider switching to {next_model} ({usage_percent:.1f}% of current limit used)")
        except ValueError:
            pass
            
    def has_sufficient_tokens(self, model: str, estimated_tokens: Optional[int] = None) -> bool:
        """
        Check if we have sufficient tokens remaining.
        
        Args:
            model: The model to check
            estimated_tokens: Optional estimate of tokens needed. If not provided, will just check against minimum threshold.
            
        Returns:
            bool: True if we have sufficient tokens, False otherwise
        """
        try:
            group_name, limit = self._get_model_group(model)
            group_info = self.model_groups[group_name]
            min_percent = group_info["min_tokens_percent"]
            
            data = self._load_data()
            current_usage = data["model_groups"][group_name]["daily_tokens"]
            remaining_tokens = limit - current_usage
            remaining_percent = (remaining_tokens / limit) * 100
            
            # If estimated tokens provided, check if we have enough
            if estimated_tokens is not None:
                if estimated_tokens > remaining_tokens:
                    log.error(f"⚠️  Insufficient tokens for estimated usage:")
                    log.error(f"  Estimated tokens needed: {estimated_tokens:,}")
                    log.error(f"  Remaining tokens: {remaining_tokens:,}")
                    return False
            
            # Check if we're below minimum threshold
            if remaining_percent < min_percent:
                log.error(f"⚠️  Token usage for {group_name} is too high to start new request:")
                log.error(f"  Current usage: {current_usage:,}/{limit:,} tokens")
                log.error(f"  Remaining: {remaining_percent:.1f}% (minimum {min_percent}% required)")
                log.error(f"  This affects models: {', '.join(group_info['models'])}")
                
                if self.stop_at_daily_token_limit:
                    log.error("stop_at_daily_token_limit is enabled - will not continue with paid tokens")
                else:
                    log.warning("stop_at_daily_token_limit is disabled - will continue with paid tokens if needed")
                return not self.stop_at_daily_token_limit
            
            return True
            
        except ValueError:
            log.warning(f"No limit configured for model {model} - allowing usage")
            return True

    def _ensure_request_tracking(self) -> None:
        """Ensure request tracking data exists"""
        data = self._load_data()
        if "request_tracking" not in data:
            data["request_tracking"] = {
                "gemini": {
                    "minute": {
                        "count": 0,
                        "timestamp": self._get_current_minute()
                    },
                    "daily": {
                        "count": 0,
                        "last_update": self._get_current_date()
                    }
                }
            }
            self._save_data(data)
            
    def _get_current_minute(self) -> str:
        """Get current minute timestamp in configured timezone."""
        return datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M")
        
    def check_request_limit(self, model: str) -> bool:
        """Check if we've hit request limits for Gemini"""
        if not model.startswith("gemini/"):
            return False
            
        data = self._load_data()
        request_data = data["request_tracking"]["gemini"]
        
        # Check if we need to reset minute counter
        current_minute = self._get_current_minute()
        if current_minute != request_data["minute"]["timestamp"]:
            request_data["minute"] = {
                "count": 0,
                "timestamp": current_minute
            }
        
        # Check if we need to reset daily counter
        current_date = self._get_current_date()
        if current_date != request_data["daily"]["last_update"]:
            request_data["daily"] = {
                "count": 0,
                "last_update": current_date
            }
        
        # Get limits from config
        group_name = next(name for name, info in self.model_groups.items() 
                         if model in info["models"])
        limits = self.model_groups[group_name]["request_limits"]
        
        # Check limits
        if request_data["minute"]["count"] >= limits["requests_per_minute"]:
            log.warning(f"⚠️  Gemini rate limit reached: {limits['requests_per_minute']} requests per minute")
            return True
            
        if request_data["daily"]["count"] >= limits["requests_per_day"]:
            log.error(f"❌ Gemini daily request limit reached: {limits['requests_per_day']} requests per day")
            return True
            
        return False
        
    def track_request(self, model: str) -> None:
        """Track a request for Gemini models"""
        if not model.startswith("gemini/"):
            return
            
        data = self._load_data()
        request_data = data["request_tracking"]["gemini"]
        
        # Update minute counter
        request_data["minute"]["count"] += 1
        # Update daily counter
        request_data["daily"]["count"] += 1
        
        self._save_data(data)

# Create a global instance (initialized with config in pipeline.py)
tracker = None 