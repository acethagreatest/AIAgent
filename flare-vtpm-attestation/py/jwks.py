import requests


def fetch_jwks(uri: str) -> dict:
    """Fetch the JWKS data from a remote endpoint."""
    response = requests.get(uri, timeout=10)
    valid_status_code = 200
    if response.status_code == valid_status_code:
        return response.json()
    msg = f"Failed to fetch JWKS: {response.status_code}"
    raise requests.exceptions.HTTPError(msg)


def get_well_known_file(
    expected_issuer: str = "https://confidentialcomputing.googleapis.com",
    well_known_path: str = "/.well-known/openid-configuration",
) -> dict:
    """Fetch JWKS URL from well known file."""
    response = requests.get(expected_issuer + well_known_path, timeout=10)
    valid_status_code = 200
    if response.status_code == valid_status_code:
        return response.json()
    msg = f"Failed to fetch JWKS URI: {response.status_code}"
    raise requests.exceptions.HTTPError(msg)


def get_rsa_data_by_kid(kid: str) -> tuple[str, str]:
    wk = get_well_known_file()
    jwks = fetch_jwks(wk["jwks_uri"])
    for key in jwks["keys"]:
        if key["kid"] == kid:
            return key["e"], key["n"]
    raise ValueError
