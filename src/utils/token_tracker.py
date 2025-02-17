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
        self.model_tiers = config["model_tiers"]
        self.timezone = pytz.timezone(config["token_tracking"]["timezone"])
        self.stop_at_daily_token_limit = config["token_tracking"]["stop_at_daily_token_limit"]
        self.auto_switch = config["token_tracking"].get("auto_switch", False)
        self.tier_order = self.model_tiers.get("tier_order", [ "gemini_models", "openai_mini_models", "openai_large_models"])
        self.min_tokens_percent = config["token_tracking"].get("min_tokens_percent", 1)
        self.current_tier = self.model_tiers.get("tier_order", [])[0]
        self._ensure_tracking_file()
        
        # Initialize request tracking for Gemini
        self._ensure_request_tracking()
        
    def _get_current_datetime(self) -> str:
        """Get current datetime in configured timezone with ISO format."""
        return datetime.now(self.timezone).isoformat()
        
    def _get_model_group(self, model: str) -> Tuple[str, int]:
        """Get the tier name and limit for a given model."""
        for tier_name, tier_info in self.model_tiers.items():
            # Skip non-tier configuration keys
            if tier_name in ["tier_order"]:  # Skip configuration keys
                continue
            
            # Skip if not a dictionary (like tier_order list)
            if not isinstance(tier_info, dict):
                continue
            
            # Skip disabled tiers
            if not tier_info.get("enabled", True):
                continue
            
            tier_models = [
                tier_info["architect_agent"],
                tier_info["user_agent"],
                tier_info["system_agent"],
                tier_info["submission_model"]
            ]
            if model in tier_models:
                # Return daily_token_limit for OpenAI, daily_request_limit for Gemini
                limit = tier_info.get("daily_token_limit", tier_info.get("daily_request_limit", 0))
                return tier_name, limit
        raise ValueError(f"Model {model} not found in any tier")
        
    def _ensure_tracking_file(self) -> None:
        """Ensure the token tracking file exists with valid initial data."""
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.tracking_file.exists():
            # Get all valid tier names (excluding special keys)
            tier_names = [name for name, info in self.model_tiers.items() 
                         if isinstance(info, dict) and name != "tier_order"]
            
            initial_data = {
                "last_update": self._get_current_datetime(),
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
                    tier_name: {
                        "daily_tokens": 0,
                        "lifetime_tokens": 0
                    }
                    for tier_name in tier_names
                },
                "models": {},
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
        # Parse the last update time
        last_update = datetime.fromisoformat(data["last_update"])
        current_time = datetime.now(self.timezone)
        
        # Only reset if the dates are different in the configured timezone
        if last_update.astimezone(self.timezone).date() != current_time.date():
            log.info(f"Resetting daily counts (last update: {last_update}, current time: {current_time})")
            data["last_update"] = self._get_current_datetime()
            
            # Reset token counts
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
            
            # Reset Gemini request tracking
            if "request_tracking" in data and "gemini" in data["request_tracking"]:
                data["request_tracking"]["gemini"]["daily"] = {
                    "count": 0,
                    "last_update": self._get_current_datetime()
                }
                # Also reset minute tracking at day boundary
                data["request_tracking"]["gemini"]["minute"] = {
                    "count": 0,
                    "timestamp": self._get_current_datetime()
                }
                log.info("Reset Gemini request tracking counts")
        
        return data
    
    def check_limit(self, model: str, tokens: int) -> bool:
        """Check if adding tokens would exceed daily limit."""
        # For Gemini models, we don't check token limits
        if model.startswith("gemini/"):
            return False  # Let check_request_limit handle Gemini limits
        
        group_name, limit = self._get_model_group(model)
        data = self._load_data()
        
        # Initialize group if it doesn't exist
        if group_name not in data["model_groups"]:
            data["model_groups"][group_name] = {
                "daily_tokens": 0,
                "lifetime_tokens": 0
            }
            self._save_data(data)
        
        current_usage = data["model_groups"][group_name]["daily_tokens"]
        
        if current_usage + tokens > limit:
            log.error(f"⚠️  Adding {tokens} tokens would exceed daily limit of {limit} for group {group_name}")
            log.error(f"Current group usage: {current_usage} tokens")
            return True
        
        return False
    
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
        
        # Find current tier
        current_tier = None
        for tier_name, tier_info in self.model_tiers.items():
            if tier_name in ["tier_order"]:  # Skip non-tier keys
                continue
            if current_model in [
                tier_info["architect_agent"],
                tier_info["user_agent"],
                tier_info["system_agent"],
                tier_info["submission_model"]
            ]:
                current_tier = tier_name
                break
            
        if not current_tier:
            return None
        
        data = self._load_data()
        
        # Try tiers in priority order
        found_current = False
        for tier_name in self.tier_order:
            # Skip tiers until we find current one
            if tier_name == current_tier:
                found_current = True
                continue
            if not found_current:
                continue
            
            tier_info = self.model_tiers.get(tier_name)
            if not tier_info or not tier_info.get("enabled", True):
                continue
            
            # Check if tier has available capacity
            group_usage = data["model_groups"].get(tier_name, {"daily_tokens": 0})["daily_tokens"]
            limit = tier_info.get("daily_token_limit", tier_info.get("daily_request_limit", 0))
            
            if group_usage < limit:
                return tier_info["system_agent"]  # Return the system agent model from this tier
            
        return None

    def track_usage(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Track token usage for an AI call."""
        data = self._load_data()
        
        # Check if we need to reset daily counts
        current_date = self._get_current_datetime()
        if data["last_update"] != current_date:
            data = self._reset_daily_counts(data)
        
        total_tokens = prompt_tokens + completion_tokens
        provider = self._get_provider_from_model(model)
        
        # For Gemini, we only care about request counts, not tokens
        if model.startswith("gemini/"):
            # Track the request but don't add to token counts
            self.track_request(model)
            
            # Enhanced logging for Gemini
            log.info(f"\n📊 Request Usage (GEMINI):")
            log.info(f"  Model: {model}")
            request_data = data["request_tracking"]["gemini"]
            log.info(f"  Daily requests: {request_data['daily']['count']}")
            log.info(f"  Minute requests: {request_data['minute']['count']}")
            return
        
        # Rest of the token tracking logic for non-Gemini models...
        # (keep existing token tracking code)
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
            
    def has_sufficient_tokens(self, model: str) -> bool:
        """Check if there are sufficient tokens available for the model"""
        if not self.stop_at_daily_token_limit:
            return True
            
        try:
            tier_name, limit = self._get_model_group(model)
            usage = self.get_daily_usage(model)
            remaining_percent = ((limit - usage) / limit) * 100
            return remaining_percent >= self.min_tokens_percent
        except ValueError:
            log.warning(f"No limit configured for model {model} - allowing usage")
            return True

    def has_sufficient_requests(self, model: str) -> bool:
        """Check if there are sufficient API requests available (for Gemini)"""
        if not model.startswith("gemini/"):
            return True
            
        try:
            tier_name, _ = self._get_model_group(model)
            tier_info = self.model_tiers[tier_name]
            if "requests_per_minute" in tier_info:
                daily_requests = self.get_daily_requests(model)
                return daily_requests < tier_info["daily_request_limit"]
        except ValueError:
            log.warning(f"No request limits configured for model {model}")
        return True

    def get_daily_requests(self, model: str) -> int:
        """Get number of requests made today for a model"""
        today = self._get_current_datetime()
        data = self._load_data()
        return data.get("request_tracking", {}).get("gemini", {}).get("daily", {}).get("count", 0)

    def _ensure_request_tracking(self) -> None:
        """Ensure request tracking data exists"""
        data = self._load_data()
        if "request_tracking" not in data:
            data["request_tracking"] = {
                "gemini": {
                    "minute": {
                        "count": 0,
                        "timestamp": self._get_current_datetime()
                    },
                    "daily": {
                        "count": 0,
                        "last_update": self._get_current_datetime()
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
        
        # Only check daily limit - let API handle minute-based rate limiting
        for tier_name, tier_info in self.model_tiers.items():
            if tier_name in ["tier_order"]:
                continue
            if not isinstance(tier_info, dict):
                continue
            if model in [tier_info.get("architect_agent"), tier_info.get("user_agent"), 
                        tier_info.get("system_agent"), tier_info.get("submission_model")]:
                
                # Only check daily limit
                if request_data["daily"]["count"] >= tier_info.get("daily_request_limit", 0):
                    log.error(f"❌ Gemini daily request limit reached ({request_data['daily']['count']}/{tier_info.get('daily_request_limit')} requests)")
                    return True
                
                return False
                
        return False
        
    def track_request(self, model: str) -> None:
        """Track a request for Gemini models"""
        if not model.startswith("gemini/"):
            return
            
        data = self._load_data()
        request_data = data["request_tracking"]["gemini"]
        
        # Only track daily count
        request_data["daily"]["count"] += 1
        
        self._save_data(data)

    def get_daily_usage(self, model: str) -> int:
        """Get daily token usage for a model"""
        data = self._load_data()
        return data.get("models", {}).get(model, {}).get("daily_tokens", 0)

    def switch_tier(self, new_tier: str) -> None:
        """Switch to a new model tier"""
        if new_tier in self.model_tiers and new_tier != "tier_order":
            self.current_tier = new_tier
            log.info(f"Switched to tier: {new_tier}")

    def get_current_tier(self) -> str:
        """Get the current model tier"""
        return self.current_tier

# Create a global instance (initialized with config in pipeline.py)
tracker = None 