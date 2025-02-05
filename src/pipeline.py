from pathlib import Path
from typing import List, Literal, Dict
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

load_dotenv()  # This loads the .env file

class Pipeline:
    def __init__(self):
        """Initialize the pipeline with configuration"""
        with open("config.yaml") as f:
            self.config = yaml.safe_load(f)
            
        # Initialize token tracker
        global tracker
        if tracker is None:
            tracker = TokenTracker(self.config)
            log.info("🔄 Token tracker initialized")
            
        # Initialize context manager
        self.context_manager = ContextManager(
            context_dir=self.config["global"]["context_folder"],
            config=self.config
        )
        
        # Create required directories
        self._setup_directories()
        
        # Add the validation call to __init__
        self._validate_config()
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        log.info("Setting up directory structure")
        dirs = [
            self.config["global"]["context_folder"],
            self.config["global"]["unprocessed_folder"],
            self.config["global"]["curated_folder"],
            self.config["global"]["processed_folder"],
            Path(self.config["logging"]["log_file"]).parent
        ]
        for dir_path in dirs:
            log.debug(f"Creating directory: {dir_path}")
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def generate_blueprint(self) -> BlueprintSchema:
        """Generate blueprint using architect agent"""
        env_key = self.config["architect_agent"]["env_key"]
        log.info(f"Using {env_key} for architect agent")
        api_key = os.getenv(env_key)
        if not api_key:
            raise EnvironmentError(f"Missing API key: {env_key}")

        messages = [
            format_system_prompt(ARCHITECT_SYSTEM_PROMPT, topic=self.config["global"]["topic"]),
            format_user_prompt(ARCHITECT_USER_PROMPT, overview=self.config["global"]["overview"], topic=self.config["global"]["topic"])
        ]
        
        return structured_ai_response(
            messages=messages,
            response_model=BlueprintSchema,
            model=self.config["architect_agent"]["model"],
            api_key=api_key,
            token_tracker=tracker
        )

    def generate_user_queries(self, blueprint: BlueprintSchema) -> List[str]:
        """Generate multiple user queries in a single request"""
        env_key = self.config["user_agent"]["env_key"]
        log.info(f"Using {env_key} for user agent")
        api_key = os.getenv(env_key)
        if not api_key:
            raise EnvironmentError(f"Missing API key: {env_key}")

        messages = [
            format_user_prompt(
                GENERATE_USER_QUERIES_USER_PROMPT,
                num_conversations_per_batch=self.config["iteration"]["num_conversations_per_batch"],
                persona=blueprint.persona,
                scenarios="\n".join(blueprint.scenarios),
                topic=self.config["global"]["topic"]
            )
        ]

        result = structured_ai_response(
            messages=messages,
            response_model=QueriesSchema,
            model=self.config["user_agent"]["model"],
            api_key=api_key
        )
        
        return result.queries

    def generate_conversation(
        self, 
        query: str,
        blueprint: BlueprintSchema
    ) -> ConversationSchema:
        """Generate conversation response using system agent"""
        api_key = os.getenv(self.config["system_agent"]["env_key"])
        if not api_key:
            raise EnvironmentError(f"Missing API key: {self.config['system_agent']['env_key']}")

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
            model=self.config["system_agent"]["model"],
            api_key=api_key,
            token_tracker=tracker
        )

        return ConversationSchema(conversation=[initial_message, response])

    def evaluate_conversation(self, conversation: ConversationSchema) -> EvaluationSchema:
        """Evaluate conversation quality"""
        if not self.config["evaluation"]["enabled"]:
            return EvaluationSchema(score=100, pass_=True)
            
        api_key = os.getenv(self.config["evaluation"]["env_key"])
        if not api_key:
            raise EnvironmentError(f"Missing API key: {self.config['evaluation']['env_key']}")

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
            model=self.config["evaluation"]["model"],
            api_key=api_key,
            token_tracker=tracker
        )

    def continue_conversation(
        self,
        conversation: ConversationSchema,
        blueprint: BlueprintSchema,
        remaining_turns: int
    ) -> ConversationSchema:
        """Continue an existing conversation for more turns"""
        if remaining_turns <= 0:
            return conversation
        
        # Get submission API key for follow-up
        env_key = self.config["submission"]["env_key"]
        log.info(f"Using {env_key} for user follow-up")
        api_key = os.getenv(env_key)
        if not api_key:
            raise EnvironmentError(f"Missing API key: {env_key}")
        
        

        # Generate follow-up user message
        messages = [
            format_user_prompt(FOLLOWUP_USER_PROMPT, conversation=pretty_print_conversation(conversation), persona=blueprint.persona, topic=self.config["global"]["topic"]),
        ]
        
        # Get user's follow-up
        follow_up = structured_ai_response(
            messages=messages,
            response_model=MessageSchema,
            model=self.config["user_agent"]["model"],
            api_key=api_key
        )
        
        # Add user's follow-up to conversation
        conversation.conversation.append(follow_up)
        
        # Get assistant's response with context using training API
        context_chunks = self.context_manager.query_context(follow_up.content)
        messages = [msg.model_dump() for msg in conversation.conversation]
        
        # Get assistant's response using sharing API key
        submission_config = self.config["submission"]
        api_key = os.getenv(submission_config["env_key"])
        if not api_key:
            raise EnvironmentError(f"Missing API key: {submission_config['env_key']}")
        
        response = structured_ai_response(
            messages=messages,
            response_model=MessageSchema,
            model=submission_config["model"],
            api_key=api_key
        )
        
        # Add assistant's response to conversation
        conversation.conversation.append(response)
        
        # Continue recursively for remaining turns
        return self.continue_conversation(conversation, blueprint, remaining_turns - 1)

    def process_curated_conversations(self):
        """Process all curated conversations"""
        curated_dir = Path(self.config["global"]["curated_folder"])
        processed_dir = Path(self.config["global"]["processed_folder"])
        
        # Get blueprint for conversation continuation
        blueprint = self.generate_blueprint()
        
        for conv_file in curated_dir.glob("*.json"):
            try:
                # Load conversation
                with open(conv_file, 'r') as f:
                    conversation = ConversationSchema.model_validate_json(f.read())
                
                # Continue conversation if configured (this is now the submission)
                n_messages = self.config["submission"]["continue_conversation_for_n_messages"]
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

    def run(self):
        """Run the complete pipeline"""
        if tracker is None:
            raise RuntimeError("Token tracker not initialized")
        
        conversations = []
        iteration = 0
        infinite_mode = self.config["iteration"]["infinite_mode"]
        stop_at_n = self.config["iteration"]["stop_at_n_number_of_conversations"]
        
        while True:
            # Check if we should stop based on number of conversations
            if not infinite_mode and len(conversations) >= stop_at_n:
                log.info(f"✅ Reached target of {stop_at_n} conversations")
                break
            
            # Generate blueprint for this batch
            blueprint = self.generate_blueprint()
            
            # Generate user queries for this batch
            queries = self.generate_user_queries(blueprint)
            
            # Process each query in the batch
            for query in queries:
                # Check token limits before starting a new conversation
                if not tracker.has_sufficient_tokens(self.config["system_agent"]["model"]):
                    log.warning("🛑 Daily token limit reached, stopping pipeline")
                    return conversations
                
                conversation = self.generate_conversation(query, blueprint)
                evaluation = self.evaluate_conversation(conversation)
                
                if evaluation.pass_:
                    log.success(f"✅ Conversation passed evaluation, with a score of {evaluation.score}")
                    # Continue conversation if configured
                    n_messages = self.config["submission"]["continue_conversation_for_n_messages"]
                    if n_messages > 0:
                        conversation = self.continue_conversation(conversation, blueprint, n_messages)
                    
                    # Save to curated folder
                    curated_dir = Path(self.config["global"]["curated_folder"])
                    conv_file = curated_dir / f"conversation_{uuid.uuid4()}.json"
                else:
                    log.warning(f"❌ Conversation failed evaluation, with a score of {evaluation.score}")
                    # Save to unprocessed folder
                    unprocessed_dir = Path(self.config["global"]["unprocessed_folder"])
                    conv_file = unprocessed_dir / f"conversation_{uuid.uuid4()}.json"

                # Save conversation to appropriate directory
                with open(conv_file, 'w') as f:
                    messages = [msg.model_dump() for msg in conversation.conversation]
                    f.write(json.dumps(messages, indent=2))

                # Only add to conversations list if it passed evaluation
                if evaluation.pass_:
                    conversations.append(conversation)
                
                # Check if we've hit our target in non-infinite mode
                if not infinite_mode and len(conversations) >= stop_at_n:
                    break
            
            iteration += 1
            log.info(f"📈 Completed iteration {iteration}, generated {len(conversations)} conversations so far")
            
            # Break if we're not in infinite mode and have enough conversations
            if not infinite_mode and len(conversations) >= stop_at_n:
                break
        
        log.success(f"🎉 Pipeline completed with {len(conversations)} conversations")
        return conversations

    def _validate_config(self) -> None:
        """Validate the configuration file structure"""
        required_sections = [
            "global",
            "vector_db",
            "architect_agent",
            "user_agent", 
            "system_agent",
            "evaluation",
            "submission",
            "logging",
            "token_tracking"
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate model configurations
        for agent in ["architect_agent", "user_agent", "system_agent", "evaluation"]:
            agent_config = self.config[agent]
            if "model" not in agent_config or "env_key" not in agent_config:
                raise ValueError(f"Missing model or env_key in {agent} configuration")
        
        # Validate token tracking configuration
        token_config = self.config["token_tracking"]
        if "model_groups" not in token_config:
            raise ValueError("Missing model_groups in token_tracking configuration")
        
        log.info("Validating configuration...") 