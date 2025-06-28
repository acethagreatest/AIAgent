import sys
import os
import json
import requests  # if you plan to call an external API like ChatGPT
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from dotenv import load_dotenv
from web3 import Web3
from web3.types import RPCEndpoint, RPCResponse
from flare_ai_kit.agent.settings_models import AgentSettingsModel
from typing import Optional, Any, cast
from data_retrieval_ingestion import load_repo_files, compute_repo_embeddings, build_repo_index, retrieve_repo_context
from sentence_transformers import SentenceTransformer
from numpy import ndarray

# Load environment variables from .env.example and .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env.example"))
load_dotenv()
rpc_url = os.getenv("FLARE_RPC_URL")
private_key = os.getenv("WALLET_PRIVATE_KEY")
gemini_api_key = os.getenv("AGENT__GEMINI_API_KEY")
chatgpt_api_key = os.getenv("AGENT__CHATGPT_API_KEY")
x_api_key = os.getenv("AGENT__X_API_KEY")
openrouter_api_key = os.getenv("AGENT__OPENROUTER_API_KEY")

# Connect to Flare blockchain and monkey-patch extraData truncation
try:
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    original_make_request = w3.provider.make_request
    def patched_make_request(method: RPCEndpoint, params: Any) -> RPCResponse:
        response = original_make_request(method, params)
        if method in ("eth_getBlockByNumber", "eth_getBlockByHash"):
            result = response.get("result")
            if result and "extraData" in result:
                extra_data = result["extraData"]
                if extra_data and len(extra_data) > 66:
                    result["extraData"] = extra_data[:66]
        return response
        return cast(dict[str, Any], response)
    w3.provider.make_request = patched_make_request

    print(f"Web3 Connected: {w3.is_connected()}")
    if not w3.is_connected():
        raise ConnectionError(f"Failed to connect to Flare network at {rpc_url}")
    
    account = w3.eth.account.from_key(private_key)
    print(f"Account Address: {account.address}")
except Exception as e:
    print(f"Error: {e}")
    raise

# Helper to retrieve the latest block (it should now return a block with fixed extraData)
def get_latest_block_fixed():
    block = w3.eth.get_block("latest")
    return block


# Complex logic: analyze the block details, and then optionally send that info to an external API
def analyze_block(block: Any) -> str:
    tx_count = len(block.transactions) if block.transactions else 0 # type: ignore
    basic_info = f"Block {block.number} contains {tx_count} transactions.\n"
    basic_info += f"ExtraData: {block['extraData']}\n"
    basic_info += f"Timestamp: {block.timestamp}\n"
    # Further decode or compute details here as needed.
    return basic_info

# Existing evaluation function that uses ChatGPT to critique the proposed strategy
def evaluate_trading_strategy(strategy: str) -> str:
    """
    Evaluates the proposed trading strategy for feasibility, profitability, and risk.
    Returns an evaluation summary.
    """
    eval_prompt = (
        "You are a seasoned quantitative analyst. Evaluate the following trading strategy "
        "for its potential profitability, feasibility, and risk. Provide recommendations for improvement.\n\n"
        f"Trading Strategy:\n{strategy}\n\n"
        "Evaluation:"
    )
    return query_chatgpt(eval_prompt)

def propose_trading_strategy(flare_signal: str, repo_context: Any) -> str:
    """
    Proposes a trading strategy by combining the blockchain signal and repository context.
    """
    return f"Strategy based on flare signal: {flare_signal} and repository context: {repo_context}"

def deploy_strategy(strategy: str) -> str:
    """
    Dummy function to deploy the trading strategy.
    """
    print(f"Deploying strategy: {strategy}")
    return "Strategy deployed successfully"

# The complete strategy pipeline now calls evaluation as part of the process.
def execute_strategy_pipeline() -> Any:
    # Retrieve and analyze the latest block from Flare.
    latest_block = get_latest_block_fixed()
    flare_signal = analyze_block(latest_block)
    
    # Retrieve repository context data.
    repo_files: list[str] = load_repo_files("/path/to/your/rainlang/repo", file_ext=".rain")
    if not repo_files:
        print("Warning: No repository files found. Using empty repository context.")
        repo_context = ""
    else:
        # Print to verify that repo_files contains actual texts.
        print(f"Loaded {len(repo_files)} repository files.")
        repo_embeddings = compute_repo_embeddings(repo_files)
        # Build a FAISS index on the embeddings.
        index_instance = build_repo_index(repo_embeddings)
        # Retrieve relevant context. Adjust the query string as needed.
        context_list: list[str] = retrieve_repo_context(
            query="trading strategy", 
            model=SentenceTransformer("all-MiniLM-L6-v2"), 
            index=index_instance, 
            texts=repo_files
        )
        # Combine the list into a single context string.
        repo_context = "\n".join(context_list)
    
    # Propose a trading strategy by combining blockchain signal and repository context.
    strategy = propose_trading_strategy(flare_signal, repo_context)
    print("\nProposed Strategy:\n", strategy)
    
    # Evaluate the proposed strategy.
    evaluation = evaluate_trading_strategy(strategy)
    print("\nStrategy Evaluation:\n", evaluation)
    
    # Dummy deployment step.
    deployment_status = deploy_strategy(strategy)
    return deployment_status

def query_chatgpt(prompt: str) -> str:
    """
    Sends a prompt to OpenAI's Chat Completions API and returns the response.
    """
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {chatgpt_api_key}"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.5
    }    
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    
    if response.ok:
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    else:
        # Print debugging information:
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)
        return "Error: ChatGPT API call failed."
    
def query_gemini(prompt: str) -> str:
    """
    Sends a prompt to Google's Gemini API and returns the response.
    """
    api_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {gemini_api_key}"
    }
    payload = {
        "model": "gemini-2.0-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.5
    }    
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    
    if response.ok:
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    else:
        # Print debugging information:
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)
        return "Error: Gemini API call failed."
    
def query_xAI(prompt: str) -> str:
    """
    Sends a prompt to Google's Grok API and returns the response.
    """
    api_url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {x_api_key}"
    }
    payload = {
        "model": "grok-2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.5
    }    
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    
    if response.ok:
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    else:
        # Print debugging information:
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)
        return "Error: xAI API call failed."
    
    
def aggregate_model_summaries(response_chatgpt: str, response_gemini: str, response_xAI: str) -> str:
    aggregation_prompt = (
        "You are an expert analyst. Combine the following summaries from different AI models "
        "into a concise, coherent final summary:\n\n"
        f"ChatGPT Summary:\n{response_chatgpt}\n\n"
        f"Gemini Summary:\n{response_gemini}\n\n"
        f"xAI Summary:\n{response_xAI}\n\n"
        "Final Consolidated Summary:"
    )
    # Reuse the query_chatgpt function to aggregate; you could replace it with another aggregator if preferred.
    return query_chatgpt(aggregation_prompt)
    
    
# Dummy agent processing function using the analysis and external API query.
def process_block_summary(input_data: str) -> str:
    # You could either perform local summarization or
    # use an external LLM (for example, ChatGPT) to produce a summary.
    prompt = f"Summarize and analyze the following block data:\n{input_data}"
    response_chatgpt = query_chatgpt(prompt)
    response_gemini = query_gemini(prompt)
    response_xAI = query_xAI(prompt)

    # Combine the responses from different APIs
    """final_summary = (
        f"ChatGPT Summary:\n{response_chatgpt}\n\n"
        f"Gemini Summary:\n{response_gemini}\n\n"
        f"xAI Summary:\n{response_xAI}"
    )"""
    final_summary = aggregate_model_summaries(
        response_chatgpt, response_gemini, response_xAI
    )
    return final_summary

# Define the agent's task to summarize the latest block with complex logic.
def summarize_latest_block() -> Optional[str]:
    try:
        block = get_latest_block_fixed()
        analysis = analyze_block(block)
        # Now process the analysis information to generate a final summary.
        summary = process_block_summary(analysis)
        return summary
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    result = summarize_latest_block()
    print(f"Agent Output: {result}")