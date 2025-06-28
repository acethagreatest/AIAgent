// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract BigNumber {
    // Custom errors
    error DivisionByZero();
    error InvalidBignumLength();
    error NegativeNumber();

    // Constants
    uint256 constant WORD_SIZE = 32;
    uint256 constant WORD_BITS = 256;

    // Structure to represent large numbers
    struct BigNum {
        uint256[] limbs; // Little-endian representation
        bool negative; // Sign flag
    }

    // Basic arithmetic operations
    function add(BigNum memory a, BigNum memory b) public pure returns (BigNum memory) {
        uint256 maxLength = max(a.limbs.length, b.limbs.length);
        uint256[] memory result = new uint256[](maxLength + 1);
        uint256 carry = 0;

        for (uint256 i = 0; i < maxLength; i++) {
            uint256 aVal = i < a.limbs.length ? a.limbs[i] : 0;
            uint256 bVal = i < b.limbs.length ? b.limbs[i] : 0;

            uint256 sum;
            unchecked {
                sum = aVal + bVal + carry;
                carry = sum < aVal ? 1 : 0; // Check for overflow
            }
            result[i] = sum;
        }

        result[maxLength] = carry;

        return normalize(
            BigNum({
                limbs: result,
                negative: false // Simplified: assumes positive numbers
            })
        );
    }

    function sub(BigNum memory a, BigNum memory b) public pure returns (BigNum memory) {
        if (compare(a, b) < 0) {
            BigNum memory result = sub(b, a);
            result.negative = true;
            return result;
        }

        uint256[] memory result = new uint256[](a.limbs.length);
        uint256 borrow = 0;

        for (uint256 i = 0; i < a.limbs.length; i++) {
            uint256 bVal = i < b.limbs.length ? b.limbs[i] : 0;
            uint256 diff;

            unchecked {
                if (a.limbs[i] >= bVal + borrow) {
                    diff = a.limbs[i] - bVal - borrow;
                    borrow = 0;
                } else {
                    diff = ((1 << WORD_BITS) + a.limbs[i]) - bVal - borrow;
                    borrow = 1;
                }
            }

            result[i] = diff;
        }

        return normalize(BigNum({limbs: result, negative: false}));
    }

    function mul(BigNum memory a, BigNum memory b) public pure returns (BigNum memory) {
        uint256[] memory result = new uint256[](a.limbs.length + b.limbs.length);

        for (uint256 i = 0; i < a.limbs.length; i++) {
            uint256 carry = 0;
            for (uint256 j = 0; j < b.limbs.length; j++) {
                uint256 pos = i + j;

                // Multiply with carry
                uint256 product;
                uint256 overflow;

                assembly {
                    let xl := mload(add(add(a, 0x20), mul(i, 0x20)))
                    let yl := mload(add(add(b, 0x20), mul(j, 0x20)))

                    // Multiply 256-bit numbers
                    let mm := mulmod(xl, yl, not(0))
                    product := mul(xl, yl)
                    overflow := sub(mm, product)
                    if lt(mm, product) { overflow := add(overflow, 1) }
                }

                // Add to result with carry
                uint256 sum = result[pos] + product + carry;
                carry = overflow + (sum < result[pos] ? 1 : 0);
                result[pos] = sum;
            }

            if (i + b.limbs.length < result.length) {
                result[i + b.limbs.length] = carry;
            }
        }

        return normalize(BigNum({limbs: result, negative: a.negative != b.negative}));
    }

    function div(BigNum memory a, BigNum memory b)
        public
        pure
        returns (BigNum memory quotient, BigNum memory remainder)
    {
        if (isZero(b)) {
            revert DivisionByZero();
        }

        if (compare(a, b) < 0) {
            return (BigNum({limbs: new uint256[](1), negative: false}), a);
        }

        // Initialize quotient and remainder
        uint256[] memory q = new uint256[](a.limbs.length - b.limbs.length + 1);
        BigNum memory r = copyBigNum(a);

        // Perform long division
        for (uint256 i = a.limbs.length; i > 0; i--) {
            uint256 qGuess = estimateQuotient(r, b, i - 1);
            BigNum memory product = mul(BigNum({limbs: new uint256[](1), negative: false}), b);
            product.limbs[0] = qGuess;

            r = sub(r, shiftLeft(product, (i - 1) * WORD_BITS));
            q[i - 1] = qGuess;
        }

        return (normalize(BigNum({limbs: q, negative: a.negative != b.negative})), normalize(r));
    }

    // Modular arithmetic for RSA
    function modPow(BigNum memory base, BigNum memory exponent, BigNum memory modulus)
        public
        pure
        returns (BigNum memory)
    {
        if (isZero(modulus)) {
            revert DivisionByZero();
        }

        BigNum memory result = BigNum({limbs: new uint256[](1), negative: false});
        result.limbs[0] = 1;

        BigNum memory b = copyBigNum(base);

        for (uint256 i = 0; i < exponent.limbs.length * WORD_BITS; i++) {
            if ((exponent.limbs[i / WORD_BITS] >> (i % WORD_BITS)) & 1 == 1) {
                result = modMul(result, b, modulus);
            }
            b = modMul(b, b, modulus);
        }

        return result;
    }

    // Helper functions
    function normalize(BigNum memory num) internal pure returns (BigNum memory) {
        // Remove leading zeros
        uint256 i = num.limbs.length;
        while (i > 0 && num.limbs[i - 1] == 0) {
            i--;
        }

        if (i == 0) {
            // Number is zero
            uint256[] memory result = new uint256[](1);
            return BigNum({limbs: result, negative: false});
        }

        uint256[] memory result = new uint256[](i);
        for (uint256 j = 0; j < i; j++) {
            result[j] = num.limbs[j];
        }

        return BigNum({limbs: result, negative: num.negative});
    }

    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a > b ? a : b;
    }

    function estimateQuotient(BigNum memory dividend, BigNum memory divisor, uint256 position)
        internal
        pure
        returns (uint256)
    {
        // Handle edge cases
        if (position >= dividend.limbs.length) return 0;
        if (isZero(divisor)) revert DivisionByZero();

        // Get the most significant limbs for estimation
        uint256 n1 = position + 1 < dividend.limbs.length ? dividend.limbs[position + 1] : 0;
        uint256 n2 = dividend.limbs[position];
        uint256 d1 = divisor.limbs[divisor.limbs.length - 1];

        // Calculate initial guess
        uint256 guess;
        if (n1 == d1) {
            guess = type(uint256).max;
        } else {
            // Combine two limbs of dividend for better precision
            uint256 dividend_high = (n1 << 128) | (n2 >> 128);
            guess = dividend_high / (d1 + 1);
        }

        // Adjust the guess (usually only needs 2-3 iterations)
        bool needsAdjustment = true;
        while (needsAdjustment) {
            BigNum memory product = mul(BigNum({limbs: new uint256[](1), negative: false}), divisor);
            product.limbs[0] = guess;

            BigNum memory shifted_dividend = BigNum({limbs: new uint256[](2), negative: false});
            shifted_dividend.limbs[0] = n2;
            shifted_dividend.limbs[1] = n1;

            if (compare(product, shifted_dividend) > 0) {
                guess--;
            } else {
                needsAdjustment = false;
            }
        }

        return guess;
    }

    function shiftLeft(BigNum memory num, uint256 bits) internal pure returns (BigNum memory) {
        if (bits == 0 || isZero(num)) {
            return copyBigNum(num);
        }

        // Calculate new length needed
        uint256 extraWords = bits / WORD_BITS;
        uint256 remainingBits = bits % WORD_BITS;
        uint256 newLength = num.limbs.length + extraWords + (remainingBits > 0 ? 1 : 0);

        uint256[] memory result = new uint256[](newLength);

        // Handle full word shifts first
        for (uint256 i = 0; i < num.limbs.length; i++) {
            result[i + extraWords] = num.limbs[i];
        }

        // Handle remaining bits if any
        if (remainingBits > 0) {
            uint256 carry = 0;
            for (uint256 i = extraWords; i < newLength - 1; i++) {
                uint256 current = result[i];
                result[i] = (current << remainingBits) | carry;
                carry = current >> (WORD_BITS - remainingBits);
            }
            result[newLength - 1] = carry;
        }

        return normalize(BigNum({limbs: result, negative: num.negative}));
    }

    function modMul(BigNum memory a, BigNum memory b, BigNum memory m) internal pure returns (BigNum memory) {
        BigNum memory prod = mul(a, b);
        (, BigNum memory result) = div(prod, m);
        return result;
    }

    function compare(BigNum memory a, BigNum memory b) internal pure returns (int256) {
        if (a.negative != b.negative) {
            return a.negative ? int256(-1) : int256(1);
        }

        if (a.limbs.length != b.limbs.length) {
            return (a.limbs.length > b.limbs.length) ? int256(1) : int256(-1);
        }

        for (uint256 i = a.limbs.length; i > 0; i--) {
            if (a.limbs[i - 1] != b.limbs[i - 1]) {
                return a.limbs[i - 1] > b.limbs[i - 1] ? int256(1) : int256(-1);
            }
        }

        return 0;
    }

    function isZero(BigNum memory num) internal pure returns (bool) {
        for (uint256 i = 0; i < num.limbs.length; i++) {
            if (num.limbs[i] != 0) return false;
        }
        return true;
    }

    function copyBigNum(BigNum memory num) internal pure returns (BigNum memory) {
        uint256[] memory newLimbs = new uint256[](num.limbs.length);
        for (uint256 i = 0; i < num.limbs.length; i++) {
            newLimbs[i] = num.limbs[i];
        }
        return BigNum({limbs: newLimbs, negative: num.negative});
    }
}
