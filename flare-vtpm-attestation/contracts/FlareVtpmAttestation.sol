// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IAttestation} from "./interfaces/IAttestation.sol";
import {IVerification} from "./interfaces/IVerification.sol";
import {BaseQuoteConfig, Header, QuoteConfig} from "./types/Common.sol";
import {InvalidVerifier, PayloadValidationFailed, SignatureVerificationFailed} from "./types/Common.sol";
import {ParserUtils} from "./utils/ParserUtils.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title FlareVtpmAttestation
 * @dev A contract for verifying RSA-signed JWTs and registering virtual Trusted Platform Module (vTPM) attestations.
 * Allows for configuring required vTPM specifications and validating token-based attestations.
 */
contract FlareVtpmAttestation is IAttestation, Ownable {
    /// @notice Stores the vTPM configurations for each registered address
    mapping(address => QuoteConfig) public registeredQuotes;

    /// @notice Event emitted when a new vTPM quote configuration is registered
    event QuoteRegistered(address indexed sender, QuoteConfig config);

    /// @notice Event emitted when the base vTPM configuration requirements are updated
    event BaseQuoteConfigUpdated(string indexed imageDigest, string hwmodel, string swname, string iss, bool secboot);

    /// @notice The required base configuration for a vTPM to be considered valid
    BaseQuoteConfig internal requiredConfig;

    /// @notice Mapping of token types to their respective verifier contracts
    mapping(bytes => IVerification) public tokenTypeVerifiers;

    /**
     * @dev Initializes the contract, setting the deployer as the initial owner.
     */
    constructor() Ownable(msg.sender) {}

    /**
     * @dev Assigns a verifier contract to handle a specific token type.
     * @param verifier Address of the contract implementing the IVerification interface for this token type.
     */
    function setTokenTypeVerifier(address verifier) external onlyOwner {
        IVerification tokenTypeVerifier = IVerification(verifier);
        bytes memory tokenType = tokenTypeVerifier.tokenType();
        if (tokenType.length == 0) {
            revert InvalidVerifier();
        }
        tokenTypeVerifiers[tokenType] = tokenTypeVerifier;
    }

    /**
     * @dev Retrieves the registered vTPM quote configuration for a specific address.
     * @param quoteAddress Address of the vTPM owner.
     * @return QuoteConfig The configuration details associated with `quoteAddress`.
     */
    function getRegisteredQuote(address quoteAddress) external view returns (QuoteConfig memory) {
        return registeredQuotes[quoteAddress];
    }

    /**
     * @dev Updates the required base configuration parameters for vTPM verification.
     * Only the contract owner can set the configuration.
     * @param hwmodel Hardware model of the device.
     * @param swname Software name or OS associated with the vTPM.
     * @param imageDigest Digest of the image used for verification.
     * @param iss The issuer string for the vTPM.
     * @param secboot Boolean indicating whether secure boot is required.
     */
    function setBaseQuoteConfig(
        string calldata hwmodel,
        string calldata swname,
        string calldata imageDigest,
        string calldata iss,
        bool secboot
    ) external onlyOwner {
        requiredConfig = BaseQuoteConfig({
            hwmodel: bytes(hwmodel),
            swname: bytes(swname),
            imageDigest: bytes(imageDigest),
            iss: bytes(iss),
            secboot: secboot
        });

        emit BaseQuoteConfigUpdated(imageDigest, hwmodel, swname, iss, secboot);
    }

    /**
     * @dev Verifies a JWT-based attestation and, if valid, registers the token for the caller.
     * Uses the `tokenTypeVerifiers` to validate the signature and payload against the expected configuration.
     * @param header JWT header as a Base64URL-decoded byte array.
     * @param payload JWT payload as a Base64URL-decoded byte array.
     * @param signature Signature associated with the JWT.
     * @return success Boolean indicating if the attestation was successfully verified and registered.
     */
    function verifyAndAttest(bytes calldata header, bytes calldata payload, bytes calldata signature)
        external
        returns (bool success)
    {
        // Parse the JWT header to obtain the token type
        Header memory parsedHeader = parseHeader(header);

        // Retrieve the verifier based on the token type
        IVerification verifier = tokenTypeVerifiers[parsedHeader.tokenType];
        if (address(verifier) == address(0)) {
            revert InvalidVerifier();
        }

        // Verify the JWT signature
        (bool verified, bytes32 digest) = verifier.verifySignature(header, payload, signature, parsedHeader);
        if (!verified) {
            revert SignatureVerificationFailed("Signature does not match");
        }

        // Parse the JWT payload to obtain the vTPM configuration
        QuoteConfig memory payloadConfig = parsePayload(payload);

        // Validate the configuration in the payload
        validatePayload(payloadConfig);

        // Assign the verified digest to the configuration for record-keeping
        payloadConfig.digest = digest;

        // Register the vTPM attestation for the sender
        registeredQuotes[msg.sender] = payloadConfig;

        emit QuoteRegistered(msg.sender, payloadConfig);

        return true;
    }

    /**
     * @dev Parses the JWT header to extract metadata such as `tokenType` and `kid`.
     * Assumes a JSON structure with fields "kid" and "x5c" (PKI type) or "OIDC" as the default.
     * @param rawHeader Base64URL-decoded byte array representing the JWT header.
     * @return header A `Header` struct containing the parsed header information.
     */
    function parseHeader(bytes calldata rawHeader) internal pure returns (Header memory header) {
        // Extract "kid" field from the header
        header.kid = ParserUtils.extractStringValue(rawHeader, '"kid":"');
        if (ParserUtils.contains(rawHeader, bytes('"x5c":'))) {
            header.tokenType = bytes("PKI");
        } else {
            header.tokenType = bytes("OIDC");
        }
    }

    /**
     * @dev Parses the JWT payload to extract the vTPM configuration values.
     * @param rawPayload Base64URL-decoded byte array representing the JWT payload.
     * @return config A `QuoteConfig` struct with the parsed vTPM configuration values.
     */
    function parsePayload(bytes calldata rawPayload) internal pure returns (QuoteConfig memory config) {
        // Extract each field from the payload JSON
        config.exp = ParserUtils.extractUintValue(rawPayload, '"exp":');
        config.iat = ParserUtils.extractUintValue(rawPayload, '"iat":');
        config.base.iss = ParserUtils.extractStringValue(rawPayload, '"iss":"');
        config.base.secboot = ParserUtils.extractBoolValue(rawPayload, '"secboot":');
        config.base.hwmodel = ParserUtils.extractStringValue(rawPayload, '"hwmodel":"');
        config.base.swname = ParserUtils.extractStringValue(rawPayload, '"swname":"');
        config.base.imageDigest = ParserUtils.extractStringValue(rawPayload, '"image_digest":"');
    }

    /**
     * @dev Validates the parsed vTPM payload configuration against the required configuration.
     * Ensures that the configuration fields match the required values and checks the JWT's validity period.
     * @param config The vTPM configuration obtained from the JWT payload.
     */
    function validatePayload(QuoteConfig memory config) internal view {
        if (config.exp < block.timestamp) {
            revert PayloadValidationFailed("Invalid expiry time");
        }
        if (config.iat > block.timestamp) {
            revert PayloadValidationFailed("Invalid issued at time");
        }
        if (keccak256(config.base.iss) != keccak256(requiredConfig.iss)) {
            revert PayloadValidationFailed("Invalid issuer");
        }
        if (config.base.secboot != requiredConfig.secboot) {
            revert PayloadValidationFailed("Invalid 'secboot' value");
        }
        if (keccak256(config.base.hwmodel) != keccak256(requiredConfig.hwmodel)) {
            revert PayloadValidationFailed("Invalid hardware model");
        }
        if (keccak256(config.base.swname) != keccak256(requiredConfig.swname)) {
            revert PayloadValidationFailed("Invalid software name");
        }
        if (keccak256(config.base.imageDigest) != keccak256(requiredConfig.imageDigest)) {
            revert PayloadValidationFailed("Invalid image digest");
        }
    }
}
