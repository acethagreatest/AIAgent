// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./Common.sol";

/// @dev Struct representing an RSA public key
struct RSAPubKey {
    bytes e; // Exponent
    bytes n; // Modulus
}
