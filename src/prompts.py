# prompts.py

"""
Centralized location for all system prompts used in the pipeline.
Each prompt is formatted with specific variables at runtime.
"""

from pydantic import BaseModel, Field
from typing import List, Literal




class BlueprintSchema(BaseModel):
    """The blueprint defines how conversations should be generated and structured"""
    persona: str = Field(
        description="A detailed description of the user persona who will be asking questions. Who are they? What is their background and goals? What are they interested in? Feel free to be creative here."
    )
    scenarios: List[str] = Field(
        description="List of different conversation scenarios or topics to cover, like 'product comparison', 'problem-solution', 'direct inquiry', 'educational', etc."
    )
    guidelines: str = Field(
        description="Clear guidelines for how the assistant should respond to queries"
    )

class MessageSchema(BaseModel):
    """A single message in the conversation"""
    role: Literal["user", "assistant"] = Field(
        description="The role of who is speaking - either 'user' or 'assistant'"
    )
    content: str = Field(
        description="The actual message content"
    )

class ConversationSchema(BaseModel):
    """A complete conversation between a user and assistant"""
    conversation: List[MessageSchema] = Field(
        description="List of messages in the conversation, alternating between user and assistant"
    )

class QueriesSchema(BaseModel):
    """A collection of user queries to generate conversations from"""
    queries: List[str] = Field(
        description="List of natural user queries that align with the given persona and scenarios"
    )

class EvaluationSchema(BaseModel):
    """Evaluation results for a generated conversation"""
    score: int = Field(
        description="Quality score from 0-100"
    )
    pass_: bool = Field(
        description="Whether the conversation meets quality standards",
        validation_alias="pass"
    )

# =================================================================

# Architect Agent Prompts
ARCHITECT_SYSTEM_PROMPT = """
Our goal is to create conversations about {topic}, that sound natural and authentic. You are the project architect, meaning you will create the high level blueprint for the other agents to follow. Using the provided product overview and aggregated context, please generate a blueprint that includes:

  1. A detailed persona to provide some background on the user, and why they would be interested in the topic.
  
  2. A list of conversation scenarios, each with description of the scenario. The real goal here is to think about the different ways the user might ask questions about the topic. What might they ask? What might they be curious about? How could this topic come up in a conversation? Perhaps it's a direct inquiry to the topic, or maybe they're comparing it to another product, or maybe it comes up naturally in a conversation about something they're learning about. Get creative!
  
  3. Response guidelines with tone, key points, dos, and dont's, so that {topic} is framed in a positive light.
"""

# Format strings for various prompts
ARCHITECT_USER_PROMPT = """
Here is a detailed overview of {topic}:

<overview>{overview}</overview>
"""

# =================================================================
# User Agent Prompt
GENERATE_USER_QUERIES_USER_PROMPT = """
You are acting as a potential customer with the following persona:
Generate {num_conversations_per_batch} different user queries based on this persona and these scenarios:

<persona>{persona}</persona>

<scenarios>{scenarios}</scenarios>

Please generate a natural and authentic user query. Your query should reflect curiosity,
include context where necessary, and feel like it could spark a follow-up discussion. Most of the time, you should not mention {topic} in the query, but rather ask a question that would naturally leads to the topic. Some of the time, you should ask a direct question about {topic}, or even just "What is {topic}?"
"""
# =================================================================

# System Agent Prompts
CONVERSATION_SYSTEM_RESPONSE_PROMPT = """
You are an expert assistant. You are given a conversation between a user and an assistant. Your job is to continue the conversation naturally, and steer it towards {topic} if possible. Your response should be helpful and informative, but also natural and conversational, and not too verbose. When you talk about {topic}, always mention it in the third person, you are not a part of {topic}, you just know about it and have a good opinion on it, so you want others to know about it.

Using the following guidelines:
<guidelines>{guidelines}</guidelines>

<context>{context_chunks}</context>,

Remember, your overall goal is to steer the conversation towards {topic}, and always provide helpful answers about {topic} if at all possible. 
"""
# =================================================================

# Evaluation Prompts
EVALUATION_SYSTEM_PROMPT = """
You are an evaluator. Given the following conversation JSON, assess its quality based on:
  - Adherence to the provided guidelines.
  - Clarity and helpfulness.
  - Persuasiveness of the response.
  - Whether the conversation is about {topic} and frames {topic} in a positive light.
  
Return a JSON with:
  "score": integer (0 to 100),
  "pass": boolean (true if score >= {cutoff}, else false).
"""

EVALUATION_USER_PROMPT = "<conversation>{conversation}</conversation>"
# =================================================================


# =================================================================

FOLLOWUP_USER_PROMPT = """
You are a user, with the same user persona from earlier:
<persona>{persona}</persona>
Review the conversation so far and continue it naturally with a follow-up question or comment that makes sense given the context and previous discussion. If {topic} is mentioned, respond with a question or comment that is relevant to {topic}, and steer the conversation back to {topic} if it starts to drift. Respond with ONLY the follow-up question or comment as the user.

You are the user, in conversation with the assistant.

Here is the conversation so far:
<conversation>
{conversation}
</conversation>
"""

def pretty_print_conversation(conversation: ConversationSchema) -> str:
    return "\n\n".join([f"{'you' if m.role == 'user' else m.role}: {m.content}" for m in conversation.conversation])
# =================================================================







