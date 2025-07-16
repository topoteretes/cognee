#!/usr/bin/env python3
"""
Cognee Package Verification Script

This script helps users verify the integrity and authenticity of Cognee packages
by checking hashes, GPG signatures, and package metadata.

Usage:
    python verify_package.py [package_file] [--check-all] [--verbose]

Examples:
    python verify_package.py cognee-0.2.1.tar.gz
    python verify_package.py --check-all --verbose
    python verify_package.py cognee-0.2.1-py3-none-any.whl --verify-signature
"""

import os
import sys
import hashlib
import json
import argparse
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile
import tarfile


class PackageVerifier:
    """Handles package verification operations."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.github_api_base = "https://api.github.com/repos/topoteretes/cognee"
        self.github_releases_base = "https://github.com/topoteretes/cognee/releases"

    def log(self, message: str, level: str = "INFO"):
        """Log messages with different levels."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{level}] {message}")

    def calculate_hash(self, file_path: str, algorithm: str = "sha256") -> str:
        """Calculate hash of a file."""
        hash_obj = hashlib.new(algorithm)

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            self.log(f"Error calculating {algorithm} hash: {e}", "ERROR")
            return ""

    def verify_hash(self, file_path: str, expected_hash: str, algorithm: str = "sha256") -> bool:
        """Verify file hash against expected value."""
        calculated_hash = self.calculate_hash(file_path, algorithm)

        if not calculated_hash:
            return False

        match = calculated_hash.lower() == expected_hash.lower()

        if match:
            self.log(f"✓ {algorithm.upper()} hash verification PASSED", "INFO")
        else:
            self.log(f"✗ {algorithm.upper()} hash verification FAILED", "ERROR")
            self.log(f"  Expected: {expected_hash}", "ERROR")
            self.log(f"  Calculated: {calculated_hash}", "ERROR")

        return match

    def verify_gpg_signature(self, file_path: str, signature_path: str) -> bool:
        """Verify GPG signature of a file."""
        try:
            # Check if gpg is available
            subprocess.run(
                ["gpg", "--version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log("GPG not found. Please install GPG to verify signatures.", "WARNING")
            return False

        if not os.path.exists(signature_path):
            self.log(f"Signature file not found: {signature_path}", "WARNING")
            return False

        try:
            result = subprocess.run(
                ["gpg", "--verify", signature_path, file_path], capture_output=True, text=True
            )

            if result.returncode == 0:
                self.log("✓ GPG signature verification PASSED", "INFO")
                return True
            else:
                self.log("✗ GPG signature verification FAILED", "ERROR")
                self.log(f"GPG error: {result.stderr}", "ERROR")
                return False
        except Exception as e:
            self.log(f"Error verifying GPG signature: {e}", "ERROR")
            return False

    def get_latest_release_info(self) -> Optional[Dict]:
        """Get latest release information from GitHub API."""
        try:
            url = f"{self.github_api_base}/releases/latest"
            with urllib.request.urlopen(url) as response:
                return json.loads(response.read())
        except Exception as e:
            self.log(f"Error fetching release info: {e}", "ERROR")
            return None

    def download_checksum_file(
        self, release_info: Dict, checksum_type: str = "SHA256SUMS"
    ) -> Optional[str]:
        """Download checksum file from GitHub release."""
        for asset in release_info.get("assets", []):
            if asset["name"] == checksum_type:
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w+", delete=False, suffix=f".{checksum_type}"
                    ) as tmp:
                        with urllib.request.urlopen(asset["browser_download_url"]) as response:
                            tmp.write(response.read().decode("utf-8"))
                        return tmp.name
                except Exception as e:
                    self.log(f"Error downloading {checksum_type}: {e}", "ERROR")
                    return None
        return None

    def parse_checksum_file(self, checksum_file: str) -> Dict[str, str]:
        """Parse checksum file and return filename -> hash mapping."""
        checksums = {}
        try:
            with open(checksum_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) >= 2:
                            hash_value = parts[0]
                            filename = parts[1].lstrip("*")  # Remove binary mode indicator
                            checksums[filename] = hash_value
        except Exception as e:
            self.log(f"Error parsing checksum file: {e}", "ERROR")
        return checksums

    def verify_package_metadata(self, package_path: str) -> bool:
        """Verify package metadata and structure."""
        self.log(f"Verifying package metadata for: {package_path}")

        if package_path.endswith(".whl"):
            return self._verify_wheel_metadata(package_path)
        elif package_path.endswith(".tar.gz"):
            return self._verify_tarball_metadata(package_path)
        else:
            self.log(f"Unsupported package format: {package_path}", "WARNING")
            return False

    def _verify_wheel_metadata(self, wheel_path: str) -> bool:
        """Verify wheel package metadata."""
        try:
            with zipfile.ZipFile(wheel_path, "r") as wheel:
                # Check for required metadata files
                required_files = ["METADATA", "WHEEL"]
                metadata_files = [
                    f for f in wheel.namelist() if any(req in f for req in required_files)
                ]

                if not metadata_files:
                    self.log("✗ Required metadata files not found in wheel", "ERROR")
                    return False

                # Read and validate METADATA
                metadata_content = None
                for file in wheel.namelist():
                    if file.endswith("METADATA"):
                        metadata_content = wheel.read(file).decode("utf-8")
                        break

                if metadata_content:
                    if "Name: cognee" in metadata_content:
                        self.log("✓ Package metadata verification PASSED", "INFO")
                        return True
                    else:
                        self.log("✗ Package name verification FAILED", "ERROR")
                        return False

        except Exception as e:
            self.log(f"Error verifying wheel metadata: {e}", "ERROR")
            return False

        return False

    def _verify_tarball_metadata(self, tarball_path: str) -> bool:
        """Verify tarball package metadata."""
        try:
            with tarfile.open(tarball_path, "r:gz") as tar:
                # Look for PKG-INFO or pyproject.toml
                metadata_files = [
                    f for f in tar.getnames() if "PKG-INFO" in f or "pyproject.toml" in f
                ]

                if not metadata_files:
                    self.log("✗ No metadata files found in tarball", "ERROR")
                    return False

                # Check PKG-INFO if available
                for file in metadata_files:
                    if "PKG-INFO" in file:
                        member = tar.getmember(file)
                        content = tar.extractfile(member).read().decode("utf-8")
                        if "Name: cognee" in content:
                            self.log("✓ Package metadata verification PASSED", "INFO")
                            return True

        except Exception as e:
            self.log(f"Error verifying tarball metadata: {e}", "ERROR")
            return False

        return False

    def verify_package(self, package_path: str, verify_signature: bool = False) -> bool:
        """Comprehensive package verification."""
        if not os.path.exists(package_path):
            self.log(f"Package file not found: {package_path}", "ERROR")
            return False

        self.log(f"Starting verification of: {package_path}")
        verification_results = []

        # 1. Verify package metadata
        metadata_ok = self.verify_package_metadata(package_path)
        verification_results.append(metadata_ok)

        # 2. Get release info and checksums
        release_info = self.get_latest_release_info()
        if not release_info:
            self.log("Could not fetch release information", "WARNING")
            return all(verification_results)

        # 3. Download and verify checksums
        checksum_file = self.download_checksum_file(release_info, "SHA256SUMS")
        if checksum_file:
            checksums = self.parse_checksum_file(checksum_file)
            filename = os.path.basename(package_path)

            if filename in checksums:
                hash_ok = self.verify_hash(package_path, checksums[filename], "sha256")
                verification_results.append(hash_ok)
            else:
                self.log(f"No checksum found for {filename}", "WARNING")

            os.unlink(checksum_file)  # Clean up temp file

        # 4. Verify GPG signature if requested
        if verify_signature:
            signature_path = f"{package_path}.asc"
            if os.path.exists(signature_path):
                sig_ok = self.verify_gpg_signature(package_path, signature_path)
                verification_results.append(sig_ok)
            else:
                self.log(f"Signature file not found: {signature_path}", "WARNING")

        # Overall result
        all_passed = all(verification_results)
        if all_passed:
            self.log("🎉 Package verification PASSED", "INFO")
        else:
            self.log("❌ Package verification FAILED", "ERROR")

        return all_passed

    def verify_all_packages(self, directory: str = ".", verify_signature: bool = False) -> bool:
        """Verify all Cognee packages in a directory."""
        package_files = []

        for file in os.listdir(directory):
            if file.startswith("cognee") and (file.endswith(".whl") or file.endswith(".tar.gz")):
                package_files.append(os.path.join(directory, file))

        if not package_files:
            self.log("No Cognee packages found in directory", "WARNING")
            return False

        all_results = []
        for package_file in package_files:
            self.log(f"\n{'=' * 60}")
            result = self.verify_package(package_file, verify_signature)
            all_results.append(result)

        return all(all_results)


def main():
    parser = argparse.ArgumentParser(description="Verify Cognee package integrity and authenticity")
    parser.add_argument("package", nargs="?", help="Path to package file to verify")
    parser.add_argument(
        "--check-all", action="store_true", help="Verify all packages in current directory"
    )
    parser.add_argument(
        "--verify-signature", action="store_true", help="Also verify GPG signatures"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if not args.package and not args.check_all:
        parser.print_help()
        sys.exit(1)

    verifier = PackageVerifier(verbose=args.verbose)

    try:
        if args.check_all:
            success = verifier.verify_all_packages(".", args.verify_signature)
        else:
            success = verifier.verify_package(args.package, args.verify_signature)

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\nVerification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
