from pathlib import Path
from typing import List, Literal, Dict, Optional
from pydantic import BaseModel, Field
import yaml
import os
import uuid
from src.context_manager import ContextManager
from src.utils.ai import structured_ai_response, format_system_prompt, format_user_prompt
from src.utils.logger import log
from src.utils.token_tracker import TokenTracker, tracker
from src.prompts import (
    ARCHITECT_SYSTEM_PROMPT,
    CONVERSATION_SYSTEM_RESPONSE_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    FOLLOWUP_USER_PROMPT,
    ARCHITECT_USER_PROMPT,
    GENERATE_USER_QUERIES_USER_PROMPT,
    EVALUATION_USER_PROMPT,
    BlueprintSchema,
    QueriesSchema,
    ConversationSchema,
    EvaluationSchema,
    MessageSchema,
    pretty_print_conversation
)
from dotenv import load_dotenv
import json
from time import sleep
from datetime import datetime, timedelta

load_dotenv()  # This loads the .env file

class Pipeline:
    def __init__(self):
        """Initialize the pipeline with configuration"""
        with open("config.yaml") as f:
            self.config = yaml.safe_load(f)
            
        # Initialize token tracker
        global tracker
        tracker = TokenTracker(self.config)
        self.tracker = tracker
        log.info("🔄 Token tracker initialized")
        
        # Initialize context manager
        self.context_manager = ContextManager(
            context_dir=self.config["directories"]["context_folder"],
            config=self.config
        )
        
        # Create required directories
        self._setup_directories()
        
        # Add the validation call to __init__
        self._validate_config()
        
        # Track current model tier - use first tier from config
        self.current_tier = self.config["model_tiers"]["tier_order"][0]
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        log.info("Setting up directory structure")
        dirs = [
            self.config["directories"]["context_folder"],
            self.config["directories"]["unprocessed_folder"],
            self.config["directories"]["curated_folder"],
            self.config["directories"]["processed_folder"],
            Path(self.config["logging"]["log_file"]).parent
        ]
        for dir_path in dirs:
            log.debug(f"Creating directory: {dir_path}")
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def generate_blueprint(self) -> Optional[BlueprintSchema]:
        """Generate blueprint using architect agent"""
        models = self._get_current_models()
        env_key = models["env_key"]
        log.info(f"Using {env_key} for architect agent")
        api_key = self._get_api_key(env_key)

        result = structured_ai_response(
            messages=[
                format_system_prompt(ARCHITECT_SYSTEM_PROMPT, topic=self.config["global"]["topic"]),
                format_user_prompt(ARCHITECT_USER_PROMPT, overview=self.config["global"]["overview"], topic=self.config["global"]["topic"])
            ],
            response_model=BlueprintSchema,
            model=models["architect_agent"],
            api_key=api_key,
            token_tracker=self.tracker
        )
        
        if result is None:
            log.warning("🛑 Could not generate blueprint - all models exhausted")
            return None
        
        return result

    def generate_user_queries(self, blueprint: BlueprintSchema) -> List[str]:
        """Generate multiple user queries in a single request"""
        models = self._get_current_models()
        env_key = models["env_key"]
        log.info(f"Using {env_key} for user agent")
        api_key = self._get_api_key(env_key)

        messages = [
            format_user_prompt(
                GENERATE_USER_QUERIES_USER_PROMPT,
                num_conversations_per_batch=1,
                persona=blueprint.persona,
                scenarios="\n".join(blueprint.scenarios),
                topic=self.config["global"]["topic"]
            )
        ]

        result = structured_ai_response(
            messages=messages,
            response_model=QueriesSchema,
            model=models["user_agent"],
            api_key=api_key
        )
        
        return result.queries

    def generate_conversation(self, query: str, blueprint: BlueprintSchema) -> Optional[ConversationSchema]:
        """Generate conversation response using system agent"""
        models = self._get_current_models()
        env_key = models["env_key"]
        api_key = self._get_api_key(env_key)

        # Get relevant context
        context_chunks = self.context_manager.query_context(query)
        
        initial_message = MessageSchema(role="user", content=query)
        system_message = format_system_prompt(CONVERSATION_SYSTEM_RESPONSE_PROMPT, 
            guidelines=blueprint.guidelines,
            context_chunks="\n".join(context_chunks),
            topic=self.config["global"]["topic"])

        messages = [system_message, {"role": "user", "content": query}]

        response = structured_ai_response(
            messages=messages,
            response_model=MessageSchema,
            model=models["system_agent"],
            api_key=api_key,
            token_tracker=tracker
        )
        
        # Check if we got a valid response
        if response is None:
            log.error("Failed to generate AI response - token limits reached")
            return None
        
        return ConversationSchema(conversation=[initial_message, response])

    def evaluate_conversation(self, conversation: ConversationSchema) -> EvaluationSchema:
        """Evaluate conversation quality"""
        if not self.config["evaluation"]["enabled"]:
            return EvaluationSchema(score=100, pass_=True)
            
        models = self._get_current_models()
        env_key = models["env_key"]
        api_key = self._get_api_key(env_key)

        messages = [
            format_system_prompt(EVALUATION_SYSTEM_PROMPT, cutoff=self.config["evaluation"]["cutoff_score"], topic=self.config["global"]["topic"]),
            format_user_prompt(
                EVALUATION_USER_PROMPT,
                conversation=conversation.model_dump_json(),
            )
        ]
        
        return structured_ai_response(
            messages=messages,
            response_model=EvaluationSchema,
            model=models["system_agent"],
            api_key=api_key,
            token_tracker=tracker
        )

    def continue_conversation(self, conversation: ConversationSchema, blueprint: BlueprintSchema, remaining_turns: int) -> ConversationSchema:
        """Continue an existing conversation for more turns"""
        if remaining_turns <= 0:
            return conversation
        
        # Use current tier's API key and models
        models = self._get_current_models()
        env_key = models["submission_env_key"]
        log.info(f"Using {env_key} for user follow-up")
        api_key = self._get_api_key(env_key)
        
        # Generate follow-up user message
        messages = [
            format_user_prompt(FOLLOWUP_USER_PROMPT, conversation=pretty_print_conversation(conversation), persona=blueprint.persona, topic=self.config["global"]["topic"]),
        ]
        
        # Get user's follow-up
        follow_up = structured_ai_response(
            messages=messages,
            response_model=MessageSchema,
            model=models["user_agent"],
            api_key=api_key
        )
        
        # Add user's follow-up to conversation
        conversation.conversation.append(follow_up)
        
        # Get assistant's response with context
        context_chunks = self.context_manager.query_context(follow_up.content)
        messages = [msg.model_dump() for msg in conversation.conversation]
        
        response = structured_ai_response(
            messages=messages,
            response_model=MessageSchema,
            model=models["submission_model"],
            api_key=api_key
        )
        
        # Add assistant's response to conversation
        conversation.conversation.append(response)
        
        # Continue recursively for remaining turns
        return self.continue_conversation(conversation, blueprint, remaining_turns - 1)

    def process_curated_conversations(self):
        """Process all conversations in the curated folder"""
        curated_dir = Path(self.config["directories"]["curated_folder"])
        processed_dir = Path(self.config["directories"]["processed_folder"])
        
        # Get blueprint for conversation continuation
        blueprint = self.generate_blueprint()
        
        for conv_file in curated_dir.glob("*.json"):
            try:
                # Load the conversation
                with open(conv_file, 'r') as f:
                    raw_messages = json.load(f)
                
                # Convert raw messages to MessageSchema objects first
                messages = [MessageSchema(**msg) for msg in raw_messages]
                
                # Create conversation object with the correct structure
                conversation_data = {"conversation": messages}
                conversation = ConversationSchema(**conversation_data)
                
                # Continue conversation if configured
                n_messages = self.config["global"]["continue_conversation_for_n_messages"]
                if n_messages > 0:
                    conversation = self.continue_conversation(conversation, blueprint, n_messages)
                    
                    # Save updated conversation back to file
                    with open(conv_file, 'w') as f:
                        messages = [msg.model_dump() for msg in conversation.conversation]
                        f.write(json.dumps(messages, indent=2))
                
                # Move to processed folder
                processed_file = processed_dir / conv_file.name
                conv_file.rename(processed_file)
                log.success(f"Successfully processed and moved {conv_file.name}")
                
            except Exception as e:
                log.error(f"Error processing {conv_file.name}: {str(e)}")

    def _get_current_models(self) -> Dict[str, str]:
        """Get current model configuration based on tier"""
        current_tier = self.tracker.get_current_tier()  # Use tracker's tier
        return self.config["model_tiers"][current_tier]
    
    def _switch_model_tier(self) -> bool:
        """Switch to next available model tier. Returns False if no tiers available."""
        # Use tier_order from config instead of hardcoded list
        tier_order = self.config["model_tiers"]["tier_order"]
        try:
            current_index = tier_order.index(self.current_tier)
            if current_index + 1 < len(tier_order):
                self.current_tier = tier_order[current_index + 1]
                log.info(f"Switching to model tier: {self.current_tier}")
                return True
        except ValueError:
            pass
        return False

    def _has_available_tokens(self) -> bool:
        """Check if current tier has available tokens"""
        models = self._get_current_models()
        if "daily_token_limit" in models:
            return self.tracker.has_sufficient_tokens(models["system_agent"])
        elif "daily_request_limit" in models:
            return self.tracker.has_sufficient_requests(models["system_agent"])
        return False

    def run_continuous(self):
        """
        Run pipeline continuously, handling token limits and model switching.
        
        This method will:
        1. Use mini models until token limit reached
        2. Switch to full models until token limit reached
        3. Switch to Gemini models until request limit reached
        4. Wait for 24 hours when all limits reached
        5. Repeat the process
        
        The pipeline can be stopped with Ctrl+C (KeyboardInterrupt)
        """
        while True:
            try:
                # First process any curated conversations
                try:
                    self.process_curated_conversations()
                except Exception as e:
                    log.error(f"Error processing curated conversations: {str(e)}")
                    log.exception("Detailed error traceback:")

                if not self._has_available_tokens():
                    if not self._switch_model_tier():
                        # No more tiers available, wait for reset
                        wait_time = self.config["global"]["wait_time_between_runs"]
                        log.info(f"All model tiers exhausted. Waiting {wait_time/3600} hours...")
                        sleep(wait_time)
                        self.current_tier = self.config["model_tiers"]["tier_order"][0]  # Reset to first tier
                        continue

                # Generate single conversation
                blueprint = self.generate_blueprint()
                if blueprint is None:
                    # All models exhausted, wait and try again
                    wait_time = self.config["global"]["wait_time_between_runs"]
                    print("\n\n\n")
                    log.info(f"🔄 All models exhausted. Waiting {wait_time/3600} hours before trying again...")
                    sleep(wait_time)
                    # Reset to first tier
                    self.current_tier = self.config["model_tiers"]["tier_order"][0]
                    continue
                
                query = self.generate_single_query(blueprint)
                conversation = self.generate_conversation(query, blueprint)
                
                # Check if conversation generation failed
                if conversation is None:
                    # All models exhausted, wait and try again
                    wait_time = self.config["global"]["wait_time_between_runs"]
                    log.info(f"🔄 All models exhausted. Waiting {wait_time/3600} hours before trying again...")
                    sleep(wait_time)
                    # Reset to first tier
                    self.current_tier = self.config["model_tiers"]["tier_order"][0]
                    continue
                
                # Evaluate and process
                evaluation = self.evaluate_conversation(conversation)
                if evaluation.pass_:
                    log.success(f"✅ Conversation passed evaluation with score {evaluation.score}")
                    
                    # Continue conversation if configured
                    n_messages = self.config["global"]["continue_conversation_for_n_messages"]
                    if n_messages > 0:
                        conversation = self.continue_conversation(conversation, blueprint, n_messages)
                    
                    # Save to curated folder
                    self._save_conversation(conversation, passed=True)
                else:
                    log.warning(f"❌ Conversation failed evaluation with score {evaluation.score}")
                    self._save_conversation(conversation, passed=False)

            except Exception as e:
                log.error(f"Error in pipeline: {str(e)}")
                log.exception("Detailed error traceback:")
                sleep(60)  # Brief pause on error before continuing

    def generate_single_query(self, blueprint: BlueprintSchema) -> str:
        """Generate a single user query instead of batch"""
        models = self._get_current_models()
        env_key = models["env_key"]
        api_key = self._get_api_key(env_key)  # Use helper method
        
        messages = [
            format_user_prompt(
                GENERATE_USER_QUERIES_USER_PROMPT,
                num_conversations_per_batch=1,
                persona=blueprint.persona,
                scenarios="\n".join(blueprint.scenarios),
                topic=self.config["global"]["topic"]
            )
        ]

        result = structured_ai_response(
            messages=messages,
            response_model=QueriesSchema,
            model=models["user_agent"],
            api_key=api_key
        )
        
        return result.queries[0]

    def _save_conversation(self, conversation: ConversationSchema, passed: bool):
        """Save conversation to appropriate directory"""
        folder = Path(self.config["directories"]["curated_folder"] if passed 
                     else self.config["directories"]["unprocessed_folder"])
        
        # Get current model and make it file-safe
        current_model = self._get_current_models()["system_agent"]
        model_name = current_model.replace('/', '_').replace(':', '-')
        
        # Create filename with model info and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_file = folder / f"convo_{model_name}_{timestamp}.json"
        
        with open(conv_file, 'w') as f:
            messages = [msg.model_dump() for msg in conversation.conversation]
            f.write(json.dumps(messages, indent=2))

    def run(self):
        """Run the complete pipeline"""
        """Alias for run_continuous for backward compatibility"""
        return self.run_continuous()

    def _validate_config(self) -> None:
        """Validate the configuration file structure"""
        required_sections = [
            "global",
            "vector_db",
            "evaluation",
            "logging",
            "token_tracking",
            "model_tiers",
            "directories"
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate model tier configurations
        required_tier_keys = ["architect_agent", "user_agent", "system_agent", "submission_model", "env_key"]
        for tier_name, tier_info in self.config["model_tiers"].items():
            # Skip non-tier configuration keys
            if tier_name in ["tier_order", "providers_to_use"]:
                continue
            
            # Skip disabled tiers
            if not tier_info.get("enabled", True):
                continue
            
            for key in required_tier_keys:
                if key not in tier_info:
                    raise ValueError(f"Missing {key} in {tier_name} configuration")
            
            # Validate limits
            if "daily_token_limit" not in tier_info and "daily_request_limit" not in tier_info:
                raise ValueError(f"Missing token or request limit in {tier_name} configuration")
        
        log.info("Validating configuration...")

    def _get_api_key(self, env_key: str) -> str:
        """Safely get API key from environment"""
        # Get current model tier info
        tier_info = self._get_current_models()
        
        # If this is a submission model, use submission_env_key
        if env_key == tier_info.get("submission_env_key"):
            api_key = os.getenv(tier_info["submission_env_key"])
        else:
            api_key = os.getenv(tier_info["env_key"])
        
        if not api_key:
            raise EnvironmentError(f"Missing API key: {env_key}")
        return api_key 