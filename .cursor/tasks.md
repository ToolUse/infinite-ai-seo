Task List

[x] 1. Project Setup:
[x] Create a new repository and set up a Python virtual environment.
[x] Define directory structure (e.g., context/, conversations/unprocessed/, conversations/curated/, conversations/processed/, logs/).
[x] Create an initial YAML configuration file.

[x] 2. Context Ingestion & Vector DB Integration:
[x] Write a module to read markdown files from the context folder.
[x] Integrate ChromaDB:
[x] Index context documents.
[x] Implement a function to query the vector database for relevant context chunks.

[x] 3. Blueprint (Architect Agent):
[x] Implement the architect agent that reads the overview key from the YAML.
[x] Generate and output a blueprint (persona, scenario types, guidelines) as a JSON object.

[x] 4. User Query Generation (User Agent):
[x] Create a module that accepts the blueprint and generates N user queries.
[x] Save each query as a JSON file in the "unprocessed" folder.

[x] 5. Assistant Response Generation (System Agent):
[x] Develop a module that:
[x] Loads each user query.
[x] Queries the vector database for context enrichment.
[x] Constructs the system agent prompt (using guidelines and retrieved context).
[x] Generates a response (and optionally a follow-up) and saves the conversation as JSON.

[x] 6. Evaluation & Curation Module:
[x] Implement a simple evaluation function using an LLM prompt:
[x] Decide on a Boolean outcome or numeric score.
[x] Filter and move approved conversations into a "curated" folder.

[x] 7. Submission Module:
[x] Write a script that iterates over curated conversation files.
[x] Send each conversation to the target API using the specified API key.
[x] Log API responses and move processed files to the "processed" folder.

[x] 8. Configuration & Logging:
[x] Refine the YAML config file with all necessary keys (global settings, agent-specific configs, vector DB settings, evaluation thresholds).
[x] Implement logging throughout all modules (error handling and debug info).

[ ] 9. Testing & Demo Preparation:
[ ] Test each module independently.
[ ] Run end-to-end tests to ensure the pipeline flows as expected.
[ ] Prepare sample context documents and run a demo iteration.
[ ] Ensure the evaluation module filters correctly and that the submission module logs successful API responses.

[ ] 10. Documentation:
[ ] Update README with instructions on setup, configuration, and running the pipeline.
[ ] Include explanations of each configuration key and agent role (for YouTube demo clarity).
