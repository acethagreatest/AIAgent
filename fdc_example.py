#!/usr/bin/env python3
"""
Example usage of Flare Data Connector (FDC) integration with the AI Agent.

This example demonstrates how to:
1. Request external data attestation
2. Retrieve attested data from Data Availability Layer
3. Verify attestation proofs
4. Use attested data in trading strategies
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from flare_ai_kit.ecosystem.protocols.fdc import FDC, AttestationType
from flare_ai_kit.ecosystem.protocols.da_layer import DALayerClient
from flare_ai_kit.ecosystem.settings_models import EcosystemSettingsModel


async def example_fdc_usage():
    """Example of using FDC for external data attestation."""
    
    # Initialize settings
    settings = EcosystemSettingsModel(
        web3_provider_url=os.getenv("FLARE_RPC_URL"),
        account_address=os.getenv("WALLET_ADDRESS"),  # Add this to your .env
        account_private_key=os.getenv("WALLET_PRIVATE_KEY"),
        is_testnet=True  # Set to False for mainnet
    )
    
    # Initialize FDC client
    fdc_client = await FDC.create(settings)
    da_client = DALayerClient("https://da-layer.flare.network")  # Replace with actual URL
    
    print("🚀 FDC Integration Example")
    print("=" * 50)
    
    # Example 1: Request Bitcoin price attestation
    print("\n1. Requesting Bitcoin price attestation...")
    price_request = await fdc_client.create_json_api_request(
        url="https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
        jq_filter=".bitcoin.usd",
        headers={"Accept": "application/json"}
    )
    
    try:
        tx_hash = await fdc_client.request_attestation(
            attestation_type=price_request.attestation_type,
            data=price_request.data,
            expected_response_hash=price_request.expected_response_hash,
            fee=price_request.fee
        )
        print(f"✅ Price attestation requested: {tx_hash}")
    except Exception as e:
        print(f"❌ Price attestation failed: {e}")
    
    # Example 2: Request transaction verification
    print("\n2. Requesting transaction verification...")
    tx_request = await fdc_client.create_evm_transaction_request(
        tx_hash="0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        chain="ETH"
    )
    
    try:
        tx_hash = await fdc_client.request_attestation(
            attestation_type=tx_request.attestation_type,
            data=tx_request.data,
            expected_response_hash=tx_request.expected_response_hash,
            fee=tx_request.fee
        )
        print(f"✅ Transaction verification requested: {tx_hash}")
    except Exception as e:
        print(f"❌ Transaction verification failed: {e}")
    
    # Example 3: Request address validity check
    print("\n3. Requesting address validity check...")
    addr_request = await fdc_client.create_address_validity_request(
        address="0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
        chain="ETH"
    )
    
    try:
        tx_hash = await fdc_client.request_attestation(
            attestation_type=addr_request.attestation_type,
            data=addr_request.data,
            expected_response_hash=addr_request.expected_response_hash,
            fee=addr_request.fee
        )
        print(f"✅ Address validity check requested: {tx_hash}")
    except Exception as e:
        print(f"❌ Address validity check failed: {e}")
    
    # Example 4: Retrieve attested data (simulated)
    print("\n4. Retrieving attested data...")
    try:
        async with da_client:
            # This would work with actual DA Layer
            print("📡 Connecting to Data Availability Layer...")
            print("ℹ️  Note: This requires actual DA Layer implementation")
    except Exception as e:
        print(f"❌ Data retrieval failed: {e}")
    
    print("\n🎯 FDC Integration Complete!")
    print("\nNext steps:")
    print("- Wait for attestation finalization (voting round completion)")
    print("- Retrieve Merkle proofs from Data Availability Layer")
    print("- Verify proofs using FdcVerification contract")
    print("- Use attested data in your smart contracts")


async def example_trading_strategy_with_fdc():
    """Example of using FDC data in a trading strategy."""
    
    print("\n🤖 AI Trading Strategy with FDC Data")
    print("=" * 50)
    
    # This would integrate with your main agent
    print("1. Analyzing Flare blockchain data...")
    print("2. Requesting external market data attestation...")
    print("3. Processing repository context...")
    print("4. Generating AI-powered trading strategy...")
    print("5. Evaluating strategy with attested data...")
    print("6. Deploying verified strategy...")
    
    print("\n✅ Strategy generated with verified external data!")


if __name__ == "__main__":
    print("Flare Data Connector (FDC) Integration Examples")
    print("=" * 60)
    
    # Run examples
    asyncio.run(example_fdc_usage())
    asyncio.run(example_trading_strategy_with_fdc())
