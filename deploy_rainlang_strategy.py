import os
import subprocess
import json
from dotenv import load_dotenv

# Load environment variables (including your API keys) if needed.
load_dotenv()

# Path and URL for Rainlang repository (modify as needed)
RAINLANG_REPO_URL = "https://github.com/rainlanguage"  # Update if necessary
LOCAL_REPO_PATH = "/Users/anthonyfisher/Documents/DocumentsMain/Dev/flare-ai-kit/rainlanguage"

def clone_or_update_repo():
    """Clone the repo if not present; otherwise, pull the latest changes."""
    if not os.path.exists(LOCAL_REPO_PATH):
        subprocess.run(["git", "clone", RAINLANG_REPO_URL, LOCAL_REPO_PATH], check=True)
        print("Cloned Rainlang repository.")
    else:
        subprocess.run(["git", "-C", LOCAL_REPO_PATH, "pull"], check=True)
        print("Updated Rainlang repository.")

clone_or_update_repo()

# Import your retrieval & ingestion functions 
from data_retrieval_ingestion import load_repo_files, compute_repo_embeddings, build_repo_index, retrieve_repo_context

# Load Rainlang files (adjust file extension if needed)
rainlang_files = load_repo_files(LOCAL_REPO_PATH, file_ext=".rain")
if not rainlang_files:
    print("No Rainlang files found.")
else:
    # You may reuse the same embedding model as your agent; ensure it is imported.
    # Here we assume you use SentenceTransformer from your main agent or data retrieval module.
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    rainlang_embeddings = compute_repo_embeddings(rainlang_files)
    repo_index = build_repo_index(rainlang_embeddings)
    
    # Retrieve context relevant to trading strategies
    query_repo = "trading strategy"
    repo_context_list = retrieve_repo_context(query_repo, embedding_model, repo_index, rainlang_files, k=5)
    repo_context = "\n".join(repo_context_list)
    print("Relevant Rainlang Repository Context:\n", repo_context)

# Next, assume you already have your Flare network scanning functions in my_agent.py.
# We import them so that we can combine the block analysis with the repo context.
from my_agent import get_latest_block_fixed, analyze_block, query_chatgpt

def propose_trading_strategy(flare_signal: str, repo_context: str) -> str:
    """
    Combines Flare block analysis with Rainlang repo context to prompt the LLM for a trading strategy.
    """
    prompt = (
        "Given the following Flare network signal data:\n"
        f"{flare_signal}\n\n"
        "And the following context from the Rainlang repository:\n"
        f"{repo_context}\n\n"
        "Suggest a profitable trading strategy that can be implemented in Rainlang. "
        "Provide a high-level plan or pseudo-code for the strategy."
    )
    # We reuse your ChatGPT query to aggregate information.
    return query_chatgpt(prompt)

# Placeholder function; replace this with actual Raindex deployment logic.
def deploy_strategy(strategy_code: str) -> str:
    print("Deploying strategy:")
    print(strategy_code)
    # For now we simply return a success message.
    return "Strategy deployed successfully."

def execute_strategy_pipeline():
    # Retrieve and analyze the latest block from Flare
    latest_block = get_latest_block_fixed()
    flare_signal = analyze_block(latest_block)
    
    # Propose trading strategy by combining blockchain signal and repository context
    strategy = propose_trading_strategy(flare_signal, repo_context)
    print("\nProposed Strategy:\n", strategy)
    
    # Deploy the strategy (placeholder)
    deployment_status = deploy_strategy(strategy)
    return deployment_status

if __name__ == "__main__":
    status = execute_strategy_pipeline()
    print("\nDeployment Status:", status)