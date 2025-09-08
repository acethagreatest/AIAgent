import sys
import os
import json
import requests  # if you plan to call an external API like ChatGPT
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from dotenv import load_dotenv
from web3 import Web3
from web3.providers import HTTPProvider
from web3.types import RPCEndpoint, RPCResponse
from eth_account import Account
from typing import Optional, Any, cast, Dict, List
from pydantic import SecretStr, HttpUrl
from data_retrieval_ingestion import load_repo_files, compute_repo_embeddings, build_repo_index, retrieve_repo_context
from sentence_transformers import SentenceTransformer
import faiss  # type: ignore
from flare_ai_kit.ecosystem.protocols.fdc import FDC, FDCResponse
from flare_ai_kit.ecosystem.protocols.da_layer import DALayerClient
from flare_ai_kit.ecosystem.settings_models import EcosystemSettingsModel

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
w3 = Web3(HTTPProvider(rpc_url))
try:
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
    
    account = Account.from_key(private_key)
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

# FDC Integration Functions
async def request_external_data_attestation(fdc_client: FDC, data_type: str, data: Dict[str, Any]) -> Optional[str]:
    """
    Request external data attestation using FDC.
    
    Args:
        fdc_client: FDC client instance
        data_type: Type of data to attest (e.g., 'price', 'transaction', 'api_data')
        data: Data to be attested
        
    Returns:
        Request ID if successful, None otherwise
    """
    try:
        if data_type == "price":
            # Request price data attestation
            request = await fdc_client.create_json_api_request(
                url="https://api.coingecko.com/api/v3/simple/price",
                jq_filter=".bitcoin.usd",
                headers={"Accept": "application/json"}
            )
        elif data_type == "transaction":
            # Request transaction verification
            request = await fdc_client.create_evm_transaction_request(
                tx_hash=data.get("tx_hash", ""),
                chain=data.get("chain", "ETH")
            )
        elif data_type == "address":
            # Request address validity check
            request = await fdc_client.create_address_validity_request(
                address=data.get("address", ""),
                chain=data.get("chain", "ETH")
            )
        else:
            # Generic JSON API request
            request = await fdc_client.create_json_api_request(
                url=data.get("url", ""),
                jq_filter=data.get("jq_filter", "."),
                headers=data.get("headers", {})
            )
        
        # Submit the attestation request
        # Note: This needs to be implemented according to FDC contract ABI
        # The data should be properly encoded as bytes and bytes32
        tx_hash = await fdc_client.request_attestation(
            attestation_type=request.attestation_type,
            data=json.dumps(request.data).encode('utf-8'),  # Convert dict to bytes
            expected_response_hash=request.expected_response_hash.encode('utf-8')[:32],  # Convert to bytes32
            fee=request.fee
        )
        
        print(f"FDC Attestation requested: {tx_hash}")
        return tx_hash
        
    except Exception as e:
        print(f"Error requesting FDC attestation: {e}")
        return None

async def get_attested_data(da_client: DALayerClient, request_id: str) -> Optional[FDCResponse]:
    """
    Retrieve attested data from Data Availability Layer.
    
    Args:
        da_client: Data Availability Layer client
        request_id: ID of the attestation request
        
    Returns:
        FDCResponse with attested data or None
    """
    try:
        async with da_client:
            response_data = await da_client.get_attestation_response(request_id)
            merkle_proof = await da_client.get_merkle_proof(request_id)
            
            if response_data and merkle_proof:
                return FDCResponse(
                    request_id=request_id,
                    response_data=response_data,
                    merkle_proof=merkle_proof,
                    merkle_root=response_data.get("merkleRoot", "")
                )
        
        return None
        
    except Exception as e:
        print(f"Error retrieving attested data: {e}")
        return None

# Enhanced strategy pipeline with FDC integration
async def execute_enhanced_strategy_pipeline() -> Any:
    """
    Enhanced strategy pipeline that includes FDC attestation for external data.
    """
    # Initialize FDC and DA Layer clients
    ecosystem_settings = EcosystemSettingsModel(
        web3_provider_url=HttpUrl(rpc_url or "https://flare-api.flare.network"),
        account_address=account.address,
        account_private_key=SecretStr(private_key) if private_key else None,
        is_testnet=True,  # Set to False for mainnet
        web3_provider_timeout=30,
        block_explorer_url=HttpUrl("https://flare-explorer.flare.network"),
        block_explorer_timeout=30
    )
    
    fdc_client = await FDC.create(ecosystem_settings)
    # da_client = DALayerClient("https://da-layer.flare.network")  # Replace with actual DA Layer URL
    
    # Retrieve and analyze the latest block from Flare
    latest_block = get_latest_block_fixed()
    flare_signal = analyze_block(latest_block)
    
    # Request external data attestation (e.g., Bitcoin price)
    print("Requesting external data attestation...")
    price_data = {"url": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd", "jq_filter": ".bitcoin.usd"}
    attestation_tx = await request_external_data_attestation(fdc_client, "price", price_data)
    
    # Retrieve repository context data
    repo_files: list[str] = load_repo_files("/path/to/your/rainlang/repo", file_ext=".rain")
    if not repo_files:
        print("Warning: No repository files found. Using empty repository context.")
        repo_context = ""
    else:
        print(f"Loaded {len(repo_files)} repository files.")
        repo_embeddings = compute_repo_embeddings(repo_files)
        index_instance: faiss.IndexFlatL2 = build_repo_index(repo_embeddings)
        context_list: list[str] = retrieve_repo_context(
            query="trading strategy", 
            model=SentenceTransformer("all-MiniLM-L6-v2"), 
            index=index_instance, 
            texts=repo_files
        )
        repo_context = "\n".join(context_list)
    
    # Enhanced strategy proposal with FDC data
    enhanced_strategy = f"""
    Strategy based on:
    - Flare blockchain signal: {flare_signal}
    - Repository context: {repo_context}
    - External data attestation: {attestation_tx if attestation_tx else 'Pending'}
    """
    
    print("\nEnhanced Strategy:\n", enhanced_strategy)
    
    # Evaluate the proposed strategy
    evaluation = evaluate_trading_strategy(enhanced_strategy)
    print("\nStrategy Evaluation:\n", evaluation)
    
    # Deploy strategy
    deployment_status = deploy_strategy(enhanced_strategy)
    return deployment_status

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
        index_instance: faiss.IndexFlatL2 = build_repo_index(repo_embeddings)
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

async def run_sports_betting_analysis():
    """Run sports betting analysis with granular analytics."""
    print("\n🏈 Running Sports Betting Analysis...")
    
    try:
        # Import sports betting components
        from sports_betting_aggregator import SportsBettingAggregator
        
        # Create aggregator
        aggregator = SportsBettingAggregator()
        
        # Initialize aggregator
        print("🔧 Initializing aggregator...")
        await aggregator.initialize()
        
        # Collect sports data
        print("📊 Collecting sports data...")
        from src.flare_ai_kit.sports.models import Sport
        data = await aggregator.collect_sports_data([Sport.NBA, Sport.NFL])
        
        # Generate sample historical data for granular analysis
        print("📈 Generating historical data for granular analysis...")
        historical_data = [
            {
                "player_id": "lebron_james",
                "player_name": "LeBron James",
                "date": "2024-01-15",
                "home_team": "Lakers",
                "away_team": "Warriors",
                "opponent_team": "Warriors",
                "venue": "Crypto.com Arena",
                "is_home": True,
                "points": 28.5,
                "rebounds": 8.2,
                "assists": 9.1
            },
            {
                "player_id": "lebron_james",
                "player_name": "LeBron James",
                "date": "2024-01-10",
                "home_team": "Warriors",
                "away_team": "Lakers",
                "opponent_team": "Warriors",
                "venue": "Chase Center",
                "is_home": False,
                "points": 25.3,
                "rebounds": 7.8,
                "assists": 8.5
            }
        ]
        
        # Analyze and predict with granular analysis
        recommendations = await aggregator.analyze_and_predict(
            data=data,
            historical_data=historical_data,
            use_granular_analysis=True
        )
        
        # Display recommendations
        await aggregator.display_recommendations(recommendations, top_n=5)
        
        return recommendations
        
    except Exception as e:
        print(f"❌ Sports betting analysis failed: {e}")
        return []

if __name__ == "__main__":
    import asyncio
    
    print("🚀 AI Agent - Choose Mode:")
    print("1. Enhanced Strategy Pipeline with FDC")
    print("2. Sports Betting Analysis")
    print("3. Both")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        # Run the enhanced strategy pipeline with FDC integration
        print("Running Enhanced AI Agent with FDC Integration...")
        try:
            result = asyncio.run(execute_enhanced_strategy_pipeline())
            print(f"Enhanced Agent Output: {result}")
        except Exception as e:
            print(f"Enhanced pipeline failed, falling back to basic pipeline: {e}")
            result = summarize_latest_block()
            print(f"Basic Agent Output: {result}")
    
    elif choice == "2":
        # Run sports betting analysis
        asyncio.run(run_sports_betting_analysis())
    
    elif choice == "3":
        # Run both
        print("Running Enhanced AI Agent with FDC Integration...")
        try:
            result = asyncio.run(execute_enhanced_strategy_pipeline())
            print(f"Enhanced Agent Output: {result}")
        except Exception as e:
            print(f"Enhanced pipeline failed: {e}")
        
        # Run sports betting
        asyncio.run(run_sports_betting_analysis())
    
    else:
        print("Invalid choice. Running default enhanced pipeline...")
        try:
            result = asyncio.run(execute_enhanced_strategy_pipeline())
            print(f"Enhanced Agent Output: {result}")
        except Exception as e:
            print(f"Enhanced pipeline failed, falling back to basic pipeline: {e}")
            result = summarize_latest_block()
            print(f"Basic Agent Output: {result}")