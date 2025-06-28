from __future__ import annotations

import base64
import datetime
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import jwt
from cryptography import x509
from cryptography.exceptions import InvalidKey
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from OpenSSL.crypto import X509, X509Store, X509StoreContext
from OpenSSL.crypto import Error as OpenSSLError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
REQUIRED_ALGO: Final[str] = "RS256"
REQUIRED_HASH_ALGO: Final[str] = "sha256"
REQUIRED_CERT_COUNT: Final[int] = 3


class PKIValidationError(Exception):
    """Base exception for all PKI validation errors."""


class InvalidCertificateChainError(PKIValidationError):
    """Raised when certificate chain validation fails."""


class CertificateParsingError(PKIValidationError):
    """Raised when certificate parsing fails."""


class SignatureValidationError(PKIValidationError):
    """Raised when signature validation fails."""


@dataclass(frozen=True)
class PKICertificates:
    """Immutable container for leaf, intermediate, and root certificates."""

    leaf_cert: x509.Certificate
    intermediate_cert: x509.Certificate
    root_cert: x509.Certificate


class PKIValidator:
    """Handles PKI token validation and certificate chain verification."""

    def __init__(self, stored_root_cert: x509.Certificate) -> None:
        self.stored_root_cert = stored_root_cert

    def validate_token(self, attestation_token: str) -> dict[str, Any]:
        """
        Validates the PKI token from the attestation service.

        Args:
            attestation_token: The JWT token to validate

        Returns:
            The decoded and verified JWT payload

        Raises:
            PKIValidationError: If any validation step fails
        """
        try:
            jwt_headers = jwt.get_unverified_header(attestation_token)
            self._validate_algorithm(jwt_headers)

            certificates = self._extract_and_validate_certificates(jwt_headers)
            self._validate_leaf_certificate(certificates.leaf_cert)
            self._compare_root_certificates(certificates.root_cert)
            self._verify_certificate_chain(certificates)

            return self._verify_token_signature(
                attestation_token, certificates.leaf_cert
            )

        except jwt.InvalidTokenError as e:
            msg = f"Invalid JWT token: {e}"
            raise PKIValidationError(msg) from e
        except Exception as e:
            msg = f"Unexpected error during validation: {e}"
            raise PKIValidationError(msg) from e

    def _validate_algorithm(self, headers: dict[str, Any]) -> None:
        """Validates the JWT algorithm header."""
        if headers.get("alg") != REQUIRED_ALGO:
            msg = f"Invalid algorithm: got {headers.get('alg')}, "
            "expected {REQUIRED_ALGORITHM}"
            raise SignatureValidationError(msg)

    def _extract_and_validate_certificates(
        self, headers: dict[str, Any]
    ) -> PKICertificates:
        """Extracts and validates certificates from x5c header."""
        x5c_headers = headers.get("x5c")
        if not x5c_headers:
            msg = "x5c header not present"
            raise PKIValidationError(msg)

        if len(x5c_headers) != REQUIRED_CERT_COUNT:
            msg = f"Expected {REQUIRED_CERT_COUNT} certificates, got {len(x5c_headers)}"
            raise PKIValidationError(msg)

        try:
            certs = [self._decode_certificate(cert) for cert in x5c_headers]
            return PKICertificates(certs[0], certs[1], certs[2])
        except (ValueError, TypeError) as e:
            msg = f"Failed to parse certificates: {e}"
            raise CertificateParsingError(msg) from e

    @staticmethod
    def _decode_certificate(cert_str: str) -> x509.Certificate:
        """Decodes and parses a base64 PEM certificate."""
        try:
            cleaned_cert = re.sub(
                r"-----BEGIN CERTIFICATE-----|-----END CERTIFICATE-----|\s+",
                "",
                cert_str,
            )
            cert_bytes = base64.b64decode(cleaned_cert)
            return x509.load_der_x509_certificate(cert_bytes, default_backend())
        except Exception as e:
            msg = f"Failed to decode certificate: {e}"
            raise CertificateParsingError(msg) from e

    def _validate_leaf_certificate(self, leaf_cert: x509.Certificate) -> None:
        """Validates the leaf certificate's algorithm and key type."""
        if not leaf_cert.signature_hash_algorithm:
            msg = "No signature hash algorithm found"
            raise SignatureValidationError(msg)

        if leaf_cert.signature_hash_algorithm.name != REQUIRED_HASH_ALGO:
            msg = "Invalid signature algorithm: "
            f"{leaf_cert.signature_hash_algorithm.name}"
            raise SignatureValidationError(msg)

        if not isinstance(leaf_cert.public_key(), rsa.RSAPublicKey):
            msg = "Leaf certificate must use RSA public key"
            raise SignatureValidationError(msg)

    def _compare_root_certificates(self, token_root_cert: x509.Certificate) -> None:
        """Compares token root certificate with stored root certificate."""
        try:
            fingerprint1 = hashlib.sha256(
                self.stored_root_cert.tbs_certificate_bytes
            ).digest()
            fingerprint2 = hashlib.sha256(
                token_root_cert.tbs_certificate_bytes
            ).digest()

            if fingerprint1 != fingerprint2:
                msg = "Root certificate fingerprint mismatch"
                raise PKIValidationError(msg)
        except AttributeError as e:
            msg = "Invalid certificate format"
            raise PKIValidationError(msg) from e

    def _verify_certificate_chain(self, certificates: PKICertificates) -> None:
        """Verifies the entire certificate chain."""
        self._check_certificate_validity(certificates)

        try:
            store = X509Store()
            store.add_cert(X509.from_cryptography(certificates.root_cert))
            store.add_cert(X509.from_cryptography(certificates.intermediate_cert))

            store_ctx = X509StoreContext(
                store, X509.from_cryptography(certificates.leaf_cert)
            )
            store_ctx.verify_certificate()

        except OpenSSLError as e:
            msg = f"Certificate chain verification failed: {e}"
            raise InvalidCertificateChainError(msg) from e

    def _check_certificate_validity(self, certificates: PKICertificates) -> None:
        """Checks the validity period of all certificates."""
        current_time = datetime.datetime.now(tz=datetime.UTC)

        for cert_name, cert in [
            ("Leaf", certificates.leaf_cert),
            ("Intermediate", certificates.intermediate_cert),
            ("Root", certificates.root_cert),
        ]:
            if not self._is_certificate_valid(cert, current_time):
                msg = f"{cert_name} certificate is not valid"
                raise InvalidCertificateChainError(msg)

    @staticmethod
    def _is_certificate_valid(
        cert: x509.Certificate, current_time: datetime.datetime
    ) -> bool:
        """Checks if a certificate is currently valid."""
        not_before = cert.not_valid_before_utc.replace(tzinfo=datetime.UTC)
        not_after = cert.not_valid_after_utc.replace(tzinfo=datetime.UTC)
        return not_before <= current_time <= not_after

    def _verify_token_signature(
        self, token: str, leaf_cert: x509.Certificate
    ) -> dict[str, Any]:
        """Verifies the token signature using the leaf certificate's public key."""
        try:
            public_key = leaf_cert.public_key()
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

            return jwt.decode(
                token,
                key=public_pem,
                algorithms=[REQUIRED_ALGO],
            )
        except (InvalidKey, jwt.InvalidTokenError) as e:
            msg = f"Token signature verification failed: {e}"
            raise SignatureValidationError(msg) from e


def main(token_path: Path, root_cert_path: Path) -> None:
    """Main function to demonstrate PKI token validation."""
    try:
        # Load the stored root certificate
        stored_root_cert = x509.load_pem_x509_certificate(
            root_cert_path.read_bytes(), default_backend()
        )

        # Initialize validator
        validator = PKIValidator(stored_root_cert)

        # Load and validate the PKI attestation token
        attestation_token = token_path.read_text().strip()

        verified_jwt = validator.validate_token(attestation_token)

        logger.info("JWT successfully validated")
        logger.info("Payload: %s", verified_jwt)

    except Exception:
        logger.exception("Validation failed")
        raise


if __name__ == "__main__":
    main(token_path=Path("data/pki.txt"), root_cert_path=Path("data/root_cert.pem"))
