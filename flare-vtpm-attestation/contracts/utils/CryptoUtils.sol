// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CryptoUtils {
    // Custom errors
    error InvalidRSAKey();
    error InvalidSignatureLength();
    error InvalidPadding();

    // Constants for RSA and SHA-256
    uint256 constant RSA_KEY_SIZE = 2048;
    uint256 constant SHA256_DIGEST_LENGTH = 32;

    struct RSAPublicKey {
        bytes modulus; // n
        bytes exponent; // e
    }

    struct RSASignature {
        bytes r; // RSASSA-PKCS1-v1_5 signature component
        bytes s; // Used in signature verification
    }

    // Main cryptographic functions
    function sha256Hash(bytes memory data) public pure returns (bytes32) {
        return sha256(data);
    }

    function verifyRSASignature(bytes memory message, RSASignature memory signature, RSAPublicKey memory publicKey)
        public
        pure
        returns (bool)
    {
        // 1. Hash the message with SHA-256
        bytes32 messageHash = sha256(message);

        // 2. Verify PKCS#1 v1.5 padding
        bytes memory em = rsaDecrypt(signature.r, publicKey);
        if (!verifyPKCS1v15Padding(em, messageHash)) {
            return false;
        }

        return true;
    }

    function rsaDecrypt(bytes memory ciphertext, RSAPublicKey memory publicKey) public pure returns (bytes memory) {
        // TODO: This is a placeholder for RSA decryption
        // In reality, would need bignum arithmetic implementation
        // c^e mod n

        require(ciphertext.length * 8 == RSA_KEY_SIZE, "Invalid ciphertext length");

        // Placeholder: returns the same length as input
        bytes memory result = new bytes(ciphertext.length);
        return result;
    }

    // PKCS#1 v1.5 related functions
    function verifyPKCS1v15Padding(bytes memory decrypted, bytes32 expectedHash) public pure returns (bool) {
        // PKCS#1 v1.5 padding format:
        // 00 || 01 || PS || 00 || T
        // where PS is padding string of 0xFF bytes
        // and T is the DER encoding of the hash

        if (decrypted.length < SHA256_DIGEST_LENGTH + 11) {
            return false;
        }

        // Check initial bytes
        if (decrypted[0] != 0x00 || decrypted[1] != 0x01) {
            return false;
        }

        // Find padding terminator
        uint256 paddingEnd = findPaddingEnd(decrypted);
        if (paddingEnd == 0) {
            return false;
        }

        // Verify hash
        return verifyHashInPadding(decrypted, paddingEnd, expectedHash);
    }

    // ASN.1 DER encoding functions
    function derEncodeHash(bytes32 hash) public pure returns (bytes memory) {
        // ASN.1 DER encoding for SHA-256:
        // 30 31 30 0d 06 09 60 86 48 01 65 03 04 02 01 05 00 04 20 || H
        bytes memory prefix = hex"3031300d060960864801650304020105000420";

        bytes memory result = new bytes(prefix.length + 32);

        // Copy prefix
        for (uint256 i = 0; i < prefix.length; i++) {
            result[i] = prefix[i];
        }

        // Copy hash
        for (uint256 i = 0; i < 32; i++) {
            result[prefix.length + i] = hash[i];
        }

        return result;
    }

    // Internal helper functions
    function findPaddingEnd(bytes memory data) internal pure returns (uint256) {
        for (uint256 i = 2; i < data.length - 1; i++) {
            if (data[i] == 0x00) {
                // Found padding terminator
                return i;
            }
            if (data[i] != 0xFF) {
                // Invalid padding byte
                return 0;
            }
        }
        return 0;
    }

    function verifyHashInPadding(bytes memory data, uint256 paddingEnd, bytes32 expectedHash)
        internal
        pure
        returns (bool)
    {
        // Get the DER encoded hash from the padding
        bytes memory derHash = derEncodeHash(expectedHash);

        // Verify the DER encoding matches
        if (data.length < paddingEnd + derHash.length) {
            return false;
        }

        for (uint256 i = 0; i < derHash.length; i++) {
            if (data[paddingEnd + 1 + i] != derHash[i]) {
                return false;
            }
        }

        return true;
    }

    // Base64 encoding/decoding functions
    function base64Encode(bytes memory data) public pure returns (string memory) {
        if (data.length == 0) return "";

        string memory table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        uint256 resultLength = 4 * ((data.length + 2) / 3);

        bytes memory result = new bytes(resultLength);
        uint256 resultIndex = 0;
        uint256 i = 0;

        for (; i + 3 <= data.length; i += 3) {
            // Process 3 bytes at a time
            uint256 value =
                uint256(uint8(data[i])) << 16 | uint256(uint8(data[i + 1])) << 8 | uint256(uint8(data[i + 2]));

            // Convert to 4 base64 characters
            result[resultIndex++] = bytes(table)[uint8(value >> 18)];
            result[resultIndex++] = bytes(table)[uint8((value >> 12) & 0x3F)];
            result[resultIndex++] = bytes(table)[uint8((value >> 6) & 0x3F)];
            result[resultIndex++] = bytes(table)[uint8(value & 0x3F)];
        }

        // Handle remaining bytes and padding
        if (i < data.length) {
            uint256 value = uint256(uint8(data[i])) << 16;
            if (i + 1 < data.length) {
                value |= uint256(uint8(data[i + 1])) << 8;
            }

            result[resultIndex++] = bytes(table)[uint8(value >> 18)];
            result[resultIndex++] = bytes(table)[uint8((value >> 12) & 0x3F)];

            if (i + 1 < data.length) {
                result[resultIndex++] = bytes(table)[uint8((value >> 6) & 0x3F)];
                result[resultIndex++] = bytes(table)[uint8(value & 0x3F)];
            } else {
                result[resultIndex++] = "=";
                result[resultIndex++] = "=";
            }
        }

        return string(result);
    }
}
