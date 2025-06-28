// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {BigNumber} from "../utils/BigNumber.sol";

import {CryptoUtils} from "../utils/CryptoUtils.sol";
import {JWTHandler} from "../utils/JWTHandler.sol";
import {PKICertificateParser} from "../utils/PKICertificateParser.sol";

contract PKIValidator {
    using BigNumber for BigNumber.BigNum;

    // Errors
    error InvalidSignature();
    error InvalidCertificate();

    // Events
    event ValidationSuccess(bytes32 indexed tokenHash);
    event ValidationFailure(bytes32 indexed tokenHash, string reason);

    // Structs
    struct ValidationResult {
        bool isValid;
        string message;
        bytes payload;
    }

    // Main validation function that ties everything together
    function validatePKIToken(bytes memory storedRootCertBytes, string memory attestationToken)
        public
        returns (ValidationResult memory)
    {
        // 1. Parse the stored root certificate
        Certificate memory storedRootCert = PKICertificateParser.decodeAndParseCertificate(string(storedRootCertBytes));

        // 2. Get JWT headers and validate algorithm
        JWTHeaders memory headers = JWTHandler.getUnverifiedHeader(attestationToken);
        if (keccak256(bytes(headers.alg)) != keccak256(bytes("RS256"))) {
            emit ValidationFailure(keccak256(bytes(attestationToken)), "Invalid algorithm");
            return ValidationResult({isValid: false, message: "Invalid algorithm - expected RS256", payload: ""});
        }

        // 3. Extract and validate certificates from x5c header
        PKICertificates memory certificates = PKICertificateParser.extractCertificateFromX5cHeader(headers.x5c);

        // 4. Convert certificate public key to BigNum format for RSA operations
        BigNumber.BigNum memory publicKey = convertPublicKeyToBigNum(certificates.leafCert.publicKey);

        // 5. Verify certificate chain using BigNumber RSA operations
        bool chainValid = verifyCertificateChainWithRSA(certificates, storedRootCert);
        if (!chainValid) {
            emit ValidationFailure(keccak256(bytes(attestationToken)), "Invalid certificate chain");
            return ValidationResult({isValid: false, message: "Invalid certificate chain", payload: ""});
        }

        // 6. Verify JWT signature using BigNumber RSA operations
        bool signatureValid = verifyJWTSignatureWithRSA(attestationToken, publicKey);
        if (!signatureValid) {
            emit ValidationFailure(keccak256(bytes(attestationToken)), "Invalid signature");
            return ValidationResult({isValid: false, message: "Invalid JWT signature", payload: ""});
        }

        // 7. Decode and return JWT payload
        bytes memory payload = JWTHandler.decodeJWT(attestationToken, certificates.leafCert.publicKey);

        emit ValidationSuccess(keccak256(bytes(attestationToken)));
        return ValidationResult({isValid: true, message: "Validation successful", payload: payload});
    }

    // Helper function to verify certificate chain using RSA
    function verifyCertificateChainWithRSA(PKICertificates memory certs, Certificate memory storedRoot)
        internal
        view
        returns (bool)
    {
        // 1. Verify root certificate matches stored root
        if (!compareCertificatesWithRSA(certs.rootCert, storedRoot)) {
            return false;
        }

        // 2. Verify intermediate cert is signed by root
        if (!verifyRSASignature(certs.intermediateCert, convertPublicKeyToBigNum(certs.rootCert.publicKey))) {
            return false;
        }

        // 3. Verify leaf cert is signed by intermediate
        return verifyRSASignature(certs.leafCert, convertPublicKeyToBigNum(certs.intermediateCert.publicKey));
    }

    // Helper function to verify JWT signature using RSA
    function verifyJWTSignatureWithRSA(string memory token, BigNumber.BigNum memory publicKey)
        internal
        pure
        returns (bool)
    {
        // 1. Split JWT
        JWTHandler.JWTParts memory parts = JWTHandler.splitJWT(token);

        // 2. Create signed data
        string memory signedData = string.concat(parts.header, ".", parts.payload);

        // 3. Decode signature from base64
        bytes memory signatureBytes = CryptoUtils.base64Decode(parts.signature);

        // 4. Convert signature to BigNum
        BigNumber.BigNum memory signature = bytesToBigNum(signatureBytes);

        // 5. Perform RSA verification
        BigNumber.BigNum memory modulus = getModulusFromPublicKey(publicKey);
        BigNumber.BigNum memory exponent = getExponentFromPublicKey(publicKey);

        // 6. Calculate s^e mod n
        BigNumber.BigNum memory calculated = publicKey.modPow(signature, exponent, modulus);

        // 7. Verify PKCS#1 v1.5 padding
        return CryptoUtils.verifyPKCS1v15Padding(bigNumToBytes(calculated), sha256(bytes(signedData)));
    }

    // Utility functions for BigNum conversions
    function convertPublicKeyToBigNum(bytes memory publicKey) internal pure returns (BigNumber.BigNum memory) {
        require(publicKey.length > 0, "Empty public key");
        return bytesToBigNum(publicKey);
    }

    function bytesToBigNum(bytes memory data) internal pure returns (BigNumber.BigNum memory) {
        uint256[] memory limbs = new uint256[]((data.length + 31) / 32);

        for (uint256 i = 0; i < data.length; i++) {
            limbs[i / 32] |= uint256(uint8(data[i])) << ((i % 32) * 8);
        }

        return BigNumber.BigNum({limbs: limbs, negative: false});
    }

    function bigNumToBytes(BigNumber.BigNum memory num) internal pure returns (bytes memory) {
        bytes memory result = new bytes(num.limbs.length * 32);

        for (uint256 i = 0; i < num.limbs.length; i++) {
            for (uint256 j = 0; j < 32; j++) {
                result[i * 32 + j] = bytes1(uint8(num.limbs[i] >> (j * 8)));
            }
        }

        return result;
    }
}
