// SPDX-License-Identifier: MIT
pragma solidity >=0.8.20;

/**
 * @title IAttestation
 * @dev Interface for attestation contracts that verify RSA-signed JWTs and register attestations.
 * Provides a standard function for validating and recording JWT-based attestations.
 */
interface IAttestation {
    /**
     * @notice Verifies an RSA-signed JWT and, if the verification succeeds, registers the attestation.
     * @dev This function expects the JWT header, payload, and signature to be provided as Base64URL-decoded byte arrays.
     * The function is intended to support attestation mechanisms that require secure JWT validation.
     * @param rawHeader The Base64URL-decoded JWT header, provided as a byte array.
     * @param rawPayload The Base64URL-decoded JWT payload, provided as a byte array.
     * @param rawSignature The Base64URL-decoded RSA signature of the JWT, provided as a byte array.
     * @return success A boolean value indicating whether the token was successfully verified and registered.
     */
    function verifyAndAttest(bytes calldata rawHeader, bytes calldata rawPayload, bytes calldata rawSignature)
        external
        returns (bool success);
}
