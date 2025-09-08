"""Data Availability Layer client for FDC attestation data."""

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


class DALayerClient:
    """Client for interacting with FDC Data Availability Layer."""

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the DA Layer client.

        Args:
            base_url: Base URL of the Data Availability Layer
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def get_attestation_response(
        self,
        request_id: str,
        round_number: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve attestation response from DA Layer.

        Args:
            request_id: ID of the attestation request
            round_number: Optional round number to query

        Returns:
            Attestation response data or None if not found
        """
        if not self.session:
            raise RuntimeError("DA Layer client not initialized. Use async context manager.")

        try:
            url = f"{self.base_url}/attestations/{request_id}"
            if round_number:
                url += f"?round={round_number}"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("Retrieved attestation response", request_id=request_id)
                    return data
                elif response.status == 404:
                    logger.warning("Attestation response not found", request_id=request_id)
                    return None
                else:
                    logger.error(
                        "Failed to retrieve attestation response",
                        request_id=request_id,
                        status=response.status
                    )
                    return None

        except Exception as e:
            logger.error("Error retrieving attestation response", error=str(e))
            return None

    async def get_merkle_proof(
        self,
        request_id: str,
        round_number: Optional[int] = None
    ) -> Optional[List[str]]:
        """
        Retrieve Merkle proof for an attestation.

        Args:
            request_id: ID of the attestation request
            round_number: Optional round number to query

        Returns:
            List of Merkle proof hashes or None if not found
        """
        if not self.session:
            raise RuntimeError("DA Layer client not initialized. Use async context manager.")

        try:
            url = f"{self.base_url}/proofs/{request_id}"
            if round_number:
                url += f"?round={round_number}"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    proof = data.get("merkleProof", [])
                    logger.info("Retrieved Merkle proof", request_id=request_id, proof_length=len(proof))
                    return proof
                elif response.status == 404:
                    logger.warning("Merkle proof not found", request_id=request_id)
                    return None
                else:
                    logger.error(
                        "Failed to retrieve Merkle proof",
                        request_id=request_id,
                        status=response.status
                    )
                    return None

        except Exception as e:
            logger.error("Error retrieving Merkle proof", error=str(e))
            return None

    async def get_round_info(self, round_number: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific voting round.

        Args:
            round_number: Round number to query

        Returns:
            Round information or None if not found
        """
        if not self.session:
            raise RuntimeError("DA Layer client not initialized. Use async context manager.")

        try:
            url = f"{self.base_url}/rounds/{round_number}"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("Retrieved round info", round_number=round_number)
                    return data
                elif response.status == 404:
                    logger.warning("Round info not found", round_number=round_number)
                    return None
                else:
                    logger.error(
                        "Failed to retrieve round info",
                        round_number=round_number,
                        status=response.status
                    )
                    return None

        except Exception as e:
            logger.error("Error retrieving round info", error=str(e))
            return None

    async def list_available_rounds(self) -> List[int]:
        """
        List all available voting rounds.

        Returns:
            List of available round numbers
        """
        if not self.session:
            raise RuntimeError("DA Layer client not initialized. Use async context manager.")

        try:
            url = f"{self.base_url}/rounds"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    rounds = data.get("rounds", [])
                    logger.info("Retrieved available rounds", count=len(rounds))
                    return rounds
                else:
                    logger.error("Failed to retrieve available rounds", status=response.status)
                    return []

        except Exception as e:
            logger.error("Error retrieving available rounds", error=str(e))
            return []

    async def get_merkle_root(self, round_number: int) -> Optional[str]:
        """
        Get Merkle root for a specific round.

        Args:
            round_number: Round number to query

        Returns:
            Merkle root hash or None if not found
        """
        round_info = await self.get_round_info(round_number)
        if round_info:
            return round_info.get("merkleRoot")
        return None
