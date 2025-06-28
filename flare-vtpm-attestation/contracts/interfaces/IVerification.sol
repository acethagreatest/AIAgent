// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

import {Header} from "../types/Common.sol";

/**
 * @title IVerification
 * @dev Interface for verification contracts that handle specific token types, such as verifying JWT signatures.
 * Provides functions to identify the token type and verify the signature of a token.
 */
interface IVerification {
    /**
     * @notice Returns the type of token this verifier handles (e.g., "OIDC").
     * @dev The token type is defined as an immutable value, specific to each implementation.
     * @return The token type identifier as a byte array.
     */
    function tokenType() external pure returns (bytes memory);

    /**
     * @notice Verifies the RSA signature of a token based on its header, payload, and provided signature.
     * @dev This function is intended to validate the authenticity of a token by checking its signature.
     * Expects the header and payload to be provided in a Base64URL-decoded format.
     * @param rawHeader The Base64URL-decoded JWT header as a byte array.
     * @param rawPayload The Base64URL-decoded JWT payload as a byte array.
     * @param rawSignature The RSA signature of the token, provided as a Base64URL-decoded byte array.
     * @param header The parsed header struct containing metadata, such as `kid` (Key ID) for locating the verification key.
     * @return success A boolean indicating if the signature verification was successful.
     * @return digest The SHA256 hash of the signing input, computed as SHA256(B64(header) + "." + B64(payload)).
     */
    function verifySignature(
        bytes calldata rawHeader,
        bytes calldata rawPayload,
        bytes calldata rawSignature,
        Header calldata header
    ) external view returns (bool success, bytes32 digest);
}
