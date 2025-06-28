// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @dev Struct representing a token header configuration
struct Header {
    bytes kid;
    bytes tokenType;
}

/// @dev Struct representing the base vTPM quote configuration
struct BaseQuoteConfig {
    bytes hwmodel;
    bytes swname;
    bytes imageDigest;
    bytes iss;
    bool secboot;
}

/// @dev Struct representing the full vTPM quote configuration
struct QuoteConfig {
    bytes32 digest;
    BaseQuoteConfig base;
    uint256 exp;
    uint256 iat;
}

// Custom reverts
error SignatureVerificationFailed(string errorMsg);
error PayloadValidationFailed(string errorMsg);
error InvalidVerifier();
