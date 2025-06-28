// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

library ParserUtils {
    /**
     * @dev Extracts a string value from a JSON-like byte array by locating a specific key and delimiter.
     * This function finds the starting point of the specified `key` in `data`, extracts the value until
     * the specified `delimiter`, and returns the extracted value as bytes.
     * @param data The byte array containing the JSON-like data.
     * @param key The key to search for, formatted with any required preceding characters, e.g., '":'.
     * @param delimiter The character that indicates the end of the value.
     * @return value The extracted value associated with the key, returned as a bytes array.
     */
    function extractValue(bytes calldata data, string memory key, string memory delimiter)
        internal
        pure
        returns (bytes memory value)
    {
        bytes memory keyBytes = bytes(key);
        bytes memory delimiterBytes = bytes(delimiter);

        uint256 start = indexOf(data, keyBytes);
        if (start == type(uint256).max) {
            return ""; // Key not found
        }
        start += keyBytes.length;

        uint256 end = indexOf(data[start:], delimiterBytes);
        if (end == type(uint256).max) {
            return ""; // Delimiter not found
        }
        value = data[start:start + end];
    }

    /**
     * @dev Extracts a string value associated with a specified key in the JSON-like data.
     * Uses `extractValue` with a quotation mark (`"`) as the delimiter.
     * @param data The byte array containing the JSON-like data.
     * @param key The key to locate the desired string value.
     * @return result The extracted string value as a bytes array.
     */
    function extractStringValue(bytes calldata data, string memory key) internal pure returns (bytes memory result) {
        return extractValue(data, key, '"');
    }

    /**
     * @dev Extracts a boolean value associated with a specified key in the JSON-like data.
     * Uses `extractValue` with a comma (`,`) as the delimiter, assuming boolean values are followed by commas.
     * @param data The byte array containing the JSON-like data.
     * @param key The key to locate the desired boolean value.
     * @return result The extracted boolean value.
     */
    function extractBoolValue(bytes calldata data, string memory key) internal pure returns (bool result) {
        bytes memory stringValue = extractValue(data, key, ",");
        return parseBool(stringValue);
    }

    /**
     * @dev Extracts a uint256 value associated with a specified key in the JSON-like data.
     * Uses `extractValue` with a comma (`,`) as the delimiter, assuming uint256 values are followed by commas.
     * @param data The byte array containing the JSON-like data.
     * @param key The key to locate the desired uint256 value.
     * @return result The extracted uint256 value.
     */
    function extractUintValue(bytes calldata data, string memory key) internal pure returns (uint256 result) {
        bytes memory stringValue = extractValue(data, key, ",");
        return parseUint(stringValue);
    }

    /**
     * @dev Parses a byte array representing a numerical value and converts it to a uint256.
     * The function iterates through the bytes, converting each valid digit to an integer and constructing
     * the uint256 result. Parsing stops at the first non-digit character.
     * @param numBytes The byte array containing the numeric value.
     * @return result The parsed uint256 integer.
     */
    function parseUint(bytes memory numBytes) internal pure returns (uint256 result) {
        for (uint256 i = 0; i < numBytes.length; i++) {
            uint8 c = uint8(numBytes[i]);
            if (c >= 48 && c <= 57) {
                // '0' to '9' in ASCII
                result = result * 10 + (c - 48);
            } else {
                break; // Stop parsing when a non-digit is encountered
            }
        }
    }

    /**
     * @dev Parses a boolean value from a byte array containing 'true' or 'false'.
     * Checks if the bytes match the string 'true', in which case it returns `true`.
     * If the bytes do not match 'true', it defaults to returning `false`.
     * @param boolBytes The byte array containing 'true' or 'false'.
     * @return result The parsed boolean value.
     */
    function parseBool(bytes memory boolBytes) internal pure returns (bool result) {
        if (
            boolBytes.length == 4 // Length of 'true'
                && boolBytes[0] == "t" && boolBytes[1] == "r" && boolBytes[2] == "u" && boolBytes[3] == "e"
        ) {
            return true;
        }
        return false;
    }

    /**
     * @dev Checks if a specific byte sequence (`needle`) exists within a larger byte sequence (`haystack`).
     * Returns `true` if the `needle` is found within the `haystack`.
     * @param haystack The larger byte array in which to search for `needle`.
     * @param needle The byte sequence to search for within `haystack`.
     * @return exists `true` if the `needle` is found in `haystack`, otherwise `false`.
     */
    function contains(bytes calldata haystack, bytes memory needle) internal pure returns (bool exists) {
        return indexOf(haystack, needle) != type(uint256).max;
    }

    /**
     * @dev Finds the index of the first occurrence of a byte sequence (`needle`) within another byte sequence (`haystack`).
     * Returns the index if `needle` is found, otherwise returns `uint256.max`.
     * @param haystack The byte array in which to search for `needle`.
     * @param needle The byte sequence to find within `haystack`.
     * @return index The index of the first occurrence of `needle`, or `uint256.max` if not found.
     */
    function indexOf(bytes calldata haystack, bytes memory needle) internal pure returns (uint256 index) {
        if (needle.length == 0 || haystack.length < needle.length) {
            return type(uint256).max;
        }

        for (uint256 i = 0; i <= haystack.length - needle.length; i++) {
            bool found = true;
            for (uint256 j = 0; j < needle.length; j++) {
                if (haystack[i + j] != needle[j]) {
                    found = false;
                    break;
                }
            }
            if (found) {
                return i;
            }
        }
        return type(uint256).max;
    }
}
