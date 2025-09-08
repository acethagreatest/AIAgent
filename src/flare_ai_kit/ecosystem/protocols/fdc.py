"""Interactions with Flare Data Connector (FDC)."""

import asyncio
from enum import Enum
from typing import Any, Dict, List, Optional, Self, Union

import structlog
from eth_typing import ChecksumAddress
from web3.types import TxParams

from flare_ai_kit.common import FlareTxError, load_abi
from flare_ai_kit.ecosystem.flare import Flare
from flare_ai_kit.ecosystem.settings_models import EcosystemSettingsModel

logger = structlog.get_logger(__name__)

# FDC Contract Addresses (these should be updated with actual addresses)
FDC_HUB_ADDRESS = "0x0000000000000000000000000000000000000000"  # TODO: Get actual address
FDC_VERIFICATION_ADDRESS = "0x0000000000000000000000000000000000000000"  # TODO: Get actual address
FDC_RELAY_ADDRESS = "0x0000000000000000000000000000000000000000"  # TODO: Get actual address


class AttestationType(Enum):
    """Supported FDC attestation types."""
    
    ADDRESS_VALIDITY = "AddressValidity"
    EVM_TRANSACTION = "EVMTransaction"
    JSON_API = "JsonApi"
    PAYMENT = "Payment"
    CONFIRMED_BLOCK_HEIGHT_EXISTS = "ConfirmedBlockHeightExists"
    BALANCE_DECREASING_TRANSACTION = "BalanceDecreasingTransaction"
    REFERENCED_PAYMENT_NONEXISTENCE = "ReferencedPaymentNonexistence"


class FDCRequest:
    """Represents an FDC attestation request."""
    
    def __init__(
        self,
        attestation_type: AttestationType,
        data: Dict[str, Any],
        expected_response_hash: str,
        fee: int = 0
    ):
        self.attestation_type = attestation_type
        self.data = data
        self.expected_response_hash = expected_response_hash
        self.fee = fee
        self.timestamp = None
        self.voting_round = None


class FDCResponse:
    """Represents an FDC attestation response."""
    
    def __init__(
        self,
        request_id: str,
        response_data: Dict[str, Any],
        merkle_proof: List[str],
        merkle_root: str
    ):
        self.request_id = request_id
        self.response_data = response_data
        self.merkle_proof = merkle_proof
        self.merkle_root = merkle_root


class FDC(Flare):
    """Handles interactions with the Flare Data Connector."""

    def __init__(self, settings: EcosystemSettingsModel) -> None:
        """
        Initialize the FDC client.

        Args:
            settings: Instance of EcosystemSettingsModel containing connection
                      and account details.
        """
        super().__init__(settings)
        self.fdc_hub = None
        self.fdc_verification = None
        self.fdc_relay = None

    @classmethod
    async def create(cls, settings: EcosystemSettingsModel) -> Self:
        """
        Asynchronously creates and initializes an FDC instance.

        Args:
            settings: Instance of EcosystemSettingsModel.

        Returns:
            A fully initialized FDC instance.
        """
        instance = cls(settings)
        logger.debug("Initializing FDC...")
        
        # Initialize FDC contracts
        instance.fdc_hub = instance.w3.eth.contract(
            address=instance.w3.to_checksum_address(FDC_HUB_ADDRESS),
            abi=load_abi("FdcHub"),  # TODO: Create FdcHub ABI
        )
        
        instance.fdc_verification = instance.w3.eth.contract(
            address=instance.w3.to_checksum_address(FDC_VERIFICATION_ADDRESS),
            abi=load_abi("FdcVerification"),  # TODO: Create FdcVerification ABI
        )
        
        instance.fdc_relay = instance.w3.eth.contract(
            address=instance.w3.to_checksum_address(FDC_RELAY_ADDRESS),
            abi=load_abi("FdcRelay"),  # TODO: Create FdcRelay ABI
        )
        
        logger.debug("FDC initialized")
        return instance

    async def request_attestation(
        self,
        attestation_type: AttestationType,
        data: Dict[str, Any],
        expected_response_hash: str,
        fee: int = 0
    ) -> str:
        """
        Submit an attestation request to the FDC.

        Args:
            attestation_type: Type of attestation to request
            data: Data specific to the attestation type
            expected_response_hash: Expected hash of the response (MIC)
            fee: Fee to pay for the attestation

        Returns:
            Transaction hash of the request submission

        Raises:
            FlareTxError: If the request submission fails
        """
        try:
            # Prepare the request data
            request_data = {
                "attestationType": attestation_type.value,
                "data": data,
                "expectedResponseHash": expected_response_hash,
                "fee": fee
            }
            
            # Call requestAttestation on FdcHub
            function_call = self.fdc_hub.functions.requestAttestation(
                attestation_type.value,
                data,
                expected_response_hash
            )
            
            # Build and send transaction
            tx = await self.build_transaction(function_call, self.address)
            if tx is None:
                raise FlareTxError("Failed to build transaction")
            
            # Add fee if specified
            if fee > 0:
                tx["value"] = fee
            
            tx_hash = await self.sign_and_send_transaction(tx)
            if tx_hash is None:
                raise FlareTxError("Failed to send transaction")
            
            logger.info(
                "Attestation request submitted",
                attestation_type=attestation_type.value,
                tx_hash=tx_hash
            )
            
            return tx_hash
            
        except Exception as e:
            logger.error("Failed to request attestation", error=str(e))
            raise FlareTxError(f"Failed to request attestation: {e}") from e

    async def verify_attestation(
        self,
        response_data: Dict[str, Any],
        merkle_proof: List[str],
        merkle_root: str
    ) -> bool:
        """
        Verify an attestation response using Merkle proof.

        Args:
            response_data: The attested response data
            merkle_proof: Merkle proof for verification
            merkle_root: Merkle root to verify against

        Returns:
            True if verification succeeds, False otherwise
        """
        try:
            # Call verifyProof on FdcVerification contract
            function_call = self.fdc_verification.functions.verifyProof(
                response_data,
                merkle_proof,
                merkle_root
            )
            
            # Execute the verification call
            result = await function_call.call()
            
            logger.info("Attestation verification completed", verified=result)
            return result
            
        except Exception as e:
            logger.error("Failed to verify attestation", error=str(e))
            return False

    async def get_attestation_data(
        self,
        request_id: str,
        da_layer_url: str
    ) -> Optional[FDCResponse]:
        """
        Retrieve attestation data from the Data Availability Layer.

        Args:
            request_id: ID of the attestation request
            da_layer_url: URL of the Data Availability Layer

        Returns:
            FDCResponse object with attestation data, or None if not found
        """
        try:
            # This would typically involve HTTP requests to the DA Layer
            # For now, return None as this requires DA Layer implementation
            logger.warning("DA Layer integration not yet implemented")
            return None
            
        except Exception as e:
            logger.error("Failed to get attestation data", error=str(e))
            return None

    async def create_address_validity_request(
        self,
        address: str,
        chain: str
    ) -> FDCRequest:
        """
        Create an AddressValidity attestation request.

        Args:
            address: Address to validate
            chain: Chain identifier

        Returns:
            FDCRequest object for AddressValidity
        """
        data = {
            "address": address,
            "chain": chain
        }
        
        # Calculate expected response hash (MIC)
        # This is a simplified example - actual implementation would hash the expected response
        expected_hash = f"addr_validity_{address}_{chain}"
        
        return FDCRequest(
            attestation_type=AttestationType.ADDRESS_VALIDITY,
            data=data,
            expected_response_hash=expected_hash
        )

    async def create_evm_transaction_request(
        self,
        tx_hash: str,
        chain: str
    ) -> FDCRequest:
        """
        Create an EVMTransaction attestation request.

        Args:
            tx_hash: Transaction hash to verify
            chain: Chain identifier (ETH, FLR, SGB)

        Returns:
            FDCRequest object for EVMTransaction
        """
        data = {
            "txHash": tx_hash,
            "chain": chain
        }
        
        expected_hash = f"evm_tx_{tx_hash}_{chain}"
        
        return FDCRequest(
            attestation_type=AttestationType.EVM_TRANSACTION,
            data=data,
            expected_response_hash=expected_hash
        )

    async def create_json_api_request(
        self,
        url: str,
        jq_filter: str,
        headers: Optional[Dict[str, str]] = None
    ) -> FDCRequest:
        """
        Create a JsonApi attestation request.

        Args:
            url: API URL to fetch data from
            jq_filter: JQ transformation filter
            headers: Optional HTTP headers

        Returns:
            FDCRequest object for JsonApi
        """
        data = {
            "url": url,
            "jqFilter": jq_filter,
            "headers": headers or {}
        }
        
        expected_hash = f"json_api_{url}_{jq_filter}"
        
        return FDCRequest(
            attestation_type=AttestationType.JSON_API,
            data=data,
            expected_response_hash=expected_hash
        )
