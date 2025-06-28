// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {FlareVtpmAttestation} from "../contracts/FlareVtpmAttestation.sol";
import {OidcSignatureVerification} from "../contracts/verifiers/OidcSignatureVerification.sol";
import {Script, console} from "forge-std/Script.sol";

contract FlareVtpmAttestationScript is Script {
    uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
    string hwmodel = vm.envString("HWMODEL");
    string swname = vm.envString("SWNAME");
    string imageDigest = vm.envString("IMAGE_DIGEST");
    string iss = vm.envString("ISS");
    bool secboot = vm.envBool("SECBOOT");

    FlareVtpmAttestation flareVtpm;
    OidcSignatureVerification oidcVerifier;

    function deploy() public {
        // Starting the broadcast of transactions from the deployer account
        vm.startBroadcast(deployerPrivateKey);

        // Deploy the FlareVtpmAttestation contract
        flareVtpm = new FlareVtpmAttestation();
        console.log("FlareVtpmAttestation deployed at:", address(flareVtpm));

        // Set base configuration on the deployed contract
        flareVtpm.setBaseQuoteConfig(hwmodel, swname, imageDigest, iss, secboot);

        // Log that the base configuration has been set
        console.log("Base quote configuration set with:");
        console.log("  Hardware Model:", hwmodel);
        console.log("  Software Name:", swname);
        console.log("  Image Digest:", imageDigest);
        console.log("  Issuer:", iss);
        console.log("  Secure Boot:", secboot);

        // Deploy the OidcSignatureVerification contract
        oidcVerifier = new OidcSignatureVerification();
        console.log("OidcSignatureVerification deployed at:", address(oidcVerifier));

        // Set the token type verifier to the OidcSignatureVerification contract
        flareVtpm.setTokenTypeVerifier(address(oidcVerifier));
        console.log("FlareVtpmAttestation token type verifier set to OidcSignatureVerification");

        vm.stopBroadcast();
    }
}
