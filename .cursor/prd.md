# Project Requirements Document: Multi-Agent AI SEO Bot

1. Overview

Project Name:
Multi-Agent AI SEO Bot

Objective:
Create an open source pipeline that leverages multiple AI agents (architect, user, system, and optionally evaluator) to generate and curate high-quality, persuasive conversational content. The content is intended to be submitted to an AI provider’s training API—allowing user-defined information (e.g., developer tool documentation) to be “ingested” by the model. An additional enhancement is to use a vector database (e.g., ChromaDB) to enrich the context retrieval, ensuring the system agent has relevant information for each query.

Primary Use Case:
• Users with rich documentation (or any other context) can load these into a folder.
• The system extracts the context, organizes it (via a vector database), and uses it to generate tailored conversation flows.
• These flows mimic a natural Q&A exchange that highlights the product or information, and then—after optional evaluation—the best conversations are submitted to the target API for training.

2. System Architecture & Data Flow

2.1 Pipeline Phases 1. Context Aggregation & Indexing:
• Input: A folder of markdown files (documentation, guidelines, etc.).
• Process:
• Read and combine context documents.
• Index content into a vector database (ChromaDB) for quick relevance-based queries.
• Output: A searchable vector index along with a unified context blob. 2. Blueprint Generation (Architect Agent):
• Input: Aggregated context and an “overview” prompt from the YAML config.
• Process:
• Use the overview to generate a blueprint that includes:
• A persona for the user agent.
• A list of conversation scenario types (e.g., direct query, problem–solution, testimonial).
• Clear response guidelines for the system agent.
• Output: A structured JSON blueprint that seeds subsequent stages. 3. User Query Generation (User Agent):
• Input: Blueprint details (persona and scenario list).
• Process:
• Generate a configurable number of natural user queries reflecting the given persona.
• Output: A set of JSON objects (one per query) stored in an “unprocessed” folder. 4. Context Enrichment for System Agent:
• Input: Each user query.
• Process:
• Before sending the query to the system agent, query the vector database for relevant context chunks.
• Merge these context chunks into the prompt (or pass them as additional context) for the system agent.
• Output: Enhanced query context to improve response quality. 5. Assistant Response Generation (System Agent):
• Input: The enriched query (user query + relevant context) along with guidelines.
• Process:
• Generate a persuasive and high-quality assistant response.
• Optionally include a follow-up question from the user agent to simulate a multi-turn conversation.
• Output: A complete conversation thread saved as a JSON file. 6. Evaluation & Curation (Optional):
• Input: Batch of generated conversation threads.
• Process:
• Run a simple evaluator (another LLM prompt) that scores each conversation (or returns a Boolean flag).
• Filter the conversations based on a cutoff score or Boolean outcome.
• Output: A curated set of conversation files stored in a “curated” folder. 7. Submission to API:
• Input: Curated conversation JSON files.
• Process:
• Submit each conversation to the target AI API using the appropriate API key.
• Log responses and move processed files to a “processed” folder.
• Output: API responses logged and conversation files marked as completed.

2.2 Components & Roles
• Architect Agent:
Creates a blueprint from the high-level overview and aggregated context. Generates a persona, scenario list, and guidelines.
• User Agent:
Uses the blueprint’s persona and scenarios to generate natural user queries.
• System Agent:
Crafts high-quality, persuasive assistant responses using both the provided guidelines and context retrieved from the vector database.
• Evaluator (Optional):
Reviews conversation outputs against quality criteria (score or Boolean check) and filters the best ones.
• Vector Database (ChromaDB):
Stores context chunks for efficient relevance queries to enhance the prompts for the system agent.
• Central YAML Config:
Governs all settings—paths, API keys, model names, prompts, number of queries, evaluation thresholds, etc.

3. Functional Requirements

   1. Context Ingestion & Indexing:
      • Read all markdown files from a designated folder.
      • Create a vector index using ChromaDB for fast retrieval.
      • Allow updates to the index as new documents are added.
   2. Blueprint Generation:
      • Read the “overview” key from the YAML config.
      • Generate a blueprint JSON that includes persona, conversation types, and guidelines.
   3. User Query Generation:
      • Generate a configurable number of queries.
      • Output each query as a JSON file into an “unprocessed” folder.
   4. Context Enrichment:
      • For each query, retrieve context chunks from the vector database based on relevance.
      • Integrate these chunks into the system agent prompt.
   5. Assistant Response Generation:
      • Accept the enriched query and guidelines.
      • Generate a response (with optional follow-up) and save as a JSON conversation file.
   6. Evaluation & Curation:
      • Run an evaluation step (either Boolean or scoring) against each conversation.
      • Filter and move approved conversations into a “curated” folder.
   7. Submission Module:
      • Iterate through curated conversations.
      • Submit each to the API using the configured API key.
      • Log responses and relocate JSON files to a “processed” folder.
   8. Configuration:
      • Use a YAML config file to define:
      • Global paths and settings.
      • Agent-specific prompts, model parameters, and output structure.
      • Vector DB settings (if applicable).
      • Evaluation criteria and thresholds.
      • API submission details.
   9. Logging & Error Handling:
      • Maintain detailed logs at every stage (ingestion, blueprint generation, query generation, response generation, evaluation, submission).
      • Provide graceful error handling and notification for any failed steps.

4. Non-Functional Requirements

   1. Performance:
      • The system should process batches of queries rapidly, keeping each phase efficient enough to allow a demo in near-real-time.
      • Vector DB queries must be fast enough to avoid noticeable latency.
   2. Modularity:
      • Each component (agent) should be loosely coupled to allow independent testing, updating, or replacement.
      • The configuration should allow quick tweaks (e.g., switching models, modifying prompts).
   3. Scalability:
      • Designed for iterative processing, with support for continuous mode if needed.
      • The vector database should handle scaling as more context documents are added.
   4. Simplicity & Clarity:
      • The architecture should remain as simple as possible for a two-hour build, with clear logs and modular components.
      • The YAML config should be straightforward to modify and explain in a demo video.

5. Dependencies & Environment
   • Programming Language: Python (assumed)
   • Libraries/Frameworks:
   • LLM client libraries (for calling the chosen models)
   • ChromaDB (for vector database functionality)
   • YAML parsing library (e.g., PyYAML)
   • JSON file handling
   • Logging framework
   • Environment Variables:
   • API key for submission (e.g., PRODUCTION_API_KEY)
   • Documentation:
   • Clear README outlining how to set up the environment, where to place context files, and how to run the pipeline.
