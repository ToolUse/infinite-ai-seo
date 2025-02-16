import os
from pathlib import Path
import yaml

import typer
from dotenv import load_dotenv

from src.pipeline import Pipeline
from src.utils.logger import setup_logger, log

# Load environment variables
load_dotenv()

app = typer.Typer()

def init_app():
    """Initialize the application"""
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    setup_logger(config)
    log.success("Application initialized and ready to go!")

@app.command()
def run():
    """Run the complete pipeline continuously"""
    try:
        log.info("Starting continuous pipeline run")
        pipeline = Pipeline()
        pipeline.run_continuous()
        
    except KeyboardInterrupt:
        log.info("Pipeline stopped by user")
    except Exception as e:
        log.error(f"Pipeline failed: {str(e)}")
        log.exception("Detailed error traceback:")

@app.command()
def index():
    """Index context documents"""
    try:
        log.info("🚀 Starting context indexing")
        pipeline = Pipeline()
        
        log.info("📚 Reading and indexing markdown files...")
        pipeline.context_manager.index_documents()
        
        log.success("✨ Successfully indexed all context documents")
        
    except Exception as e:
        log.error(f"❌ Indexing failed: {str(e)}")
        log.exception("Detailed error traceback:")

@app.command()
def reindex():
    """Reindex all context documents"""
    try:
        log.info("Starting context reindexing")
        pipeline = Pipeline()
        
        log.info("Clearing existing index...")
        log.info("Reading and reindexing markdown files...")
        pipeline.context_manager.reindex()
        
        log.success("Successfully reindexed all context documents")
        
    except Exception as e:
        log.error(f"Reindexing failed: {str(e)}")
        log.exception("Detailed error traceback:")

@app.command()
def submit():
    """Process and submit curated conversations"""
    try:
        log.info("Starting submission process")
        pipeline = Pipeline()
        
        log.info("Processing curated conversations...")
        pipeline.process_curated_conversations()
        
        log.success("Successfully processed and submitted all curated conversations")
        
    except Exception as e:
        log.error(f"Submission failed: {str(e)}")
        log.exception("Detailed error traceback:")

if __name__ == "__main__":
    init_app()
    app()
