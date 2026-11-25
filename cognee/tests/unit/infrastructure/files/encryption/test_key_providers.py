import base64
import json
import os

import pytest

from cognee.infrastructure.files.encryption import (
    AwsKmsKeyProvider,
    FileDecryptionError,
    FileEncryptionConfig,
    FileEncryptionConfigurationError,
    LocalKeyProvider,
    get_key_provider,
    register_file_key_provider,
    reset_key_providers,
)
from cognee.infrastructure.files.encryption.key_providers import (
    KMS_ENCRYPTION_CONTEXT,
    unwrap_data_key_cached,
)

from .conftest import fresh_key


def test_local_provider_wraps_and_unwraps():
    provider = LocalKeyProvider.from_config(FileEncryptionConfig(key=fresh_key()))

    data_key = provider.generate_data_key()

    assert len(data_key.plaintext) == 32
    assert data_key.wrapped != data_key.plaintext
    assert data_key.key_id == "1"
    assert provider.unwrap_data_key(data_key.wrapped, data_key.key_id) == data_key.plaintext


def test_local_provider_rotation_keeps_old_ids_readable():
    old, new = fresh_key(), fresh_key()
    before = LocalKeyProvider.from_config(FileEncryptionConfig(keys=json.dumps({"1": old})))
    written = before.generate_data_key()

    after = LocalKeyProvider.from_config(
        FileEncryptionConfig(keys=json.dumps({"1": old, "2": new}), active_key_id="2")
    )

    assert after.generate_data_key().key_id == "2"
    assert after.unwrap_data_key(written.wrapped, "1") == written.plaintext


def test_local_provider_refuses_unknown_or_wrong_key():
    provider = LocalKeyProvider.from_config(FileEncryptionConfig(key=fresh_key()))
    data_key = provider.generate_data_key()

    with pytest.raises(FileDecryptionError, match="not in the configured"):
        provider.unwrap_data_key(data_key.wrapped, "other")

    other = LocalKeyProvider.from_config(FileEncryptionConfig(key=fresh_key()))
    with pytest.raises(FileDecryptionError, match="differs"):
        other.unwrap_data_key(data_key.wrapped, "1")


def test_local_provider_reads_key_file(tmp_path):
    key_file = tmp_path / "kek"
    key_file.write_text(fresh_key() + "\n")
    provider = LocalKeyProvider.from_config(FileEncryptionConfig(key_file=str(key_file)))

    assert provider.active_key_id == "1"


@pytest.mark.parametrize(
    "config, match",
    [
        (FileEncryptionConfig(), "no key is configured"),
        (FileEncryptionConfig(key="not base64!"), "not valid base64"),
        (FileEncryptionConfig(key=base64.b64encode(b"short").decode()), "exactly 32 bytes"),
        (FileEncryptionConfig(keys="[1,2]"), "non-empty JSON object"),
        (FileEncryptionConfig(keys="{nope"), "not a JSON object"),
        (
            FileEncryptionConfig(keys=json.dumps({"1": fresh_key()}), active_key_id="2"),
            "not in the key ring",
        ),
        (FileEncryptionConfig(key_file="/nonexistent/kek"), "does not exist"),
    ],
)
def test_local_provider_configuration_errors_name_the_fix(config, match):
    with pytest.raises(FileEncryptionConfigurationError, match=match) as info:
        LocalKeyProvider.from_config(config)
    assert info.value.remediation or "base64" in str(info.value)


class FakeKms:
    """Enough of a boto3 KMS client to exercise the provider's contract."""

    def __init__(self):
        self.calls = []
        self.kek = os.urandom(32)

    def generate_data_key(self, **kwargs):
        self.calls.append(("generate_data_key", kwargs))
        plaintext = os.urandom(32)
        return {
            "Plaintext": plaintext,
            "CiphertextBlob": bytes(a ^ b for a, b in zip(plaintext, self.kek)),
            "KeyId": "arn:aws:kms:ca-central-1:123456789012:key/abc",
        }

    def decrypt(self, **kwargs):
        self.calls.append(("decrypt", kwargs))
        if kwargs.get("EncryptionContext") != KMS_ENCRYPTION_CONTEXT:
            raise RuntimeError("InvalidCiphertextException")
        blob = kwargs["CiphertextBlob"]
        return {"Plaintext": bytes(a ^ b for a, b in zip(blob, self.kek))}


def test_kms_provider_uses_generate_data_key_and_decrypt_with_context():
    kms = FakeKms()
    provider = AwsKmsKeyProvider("alias/cognee-files", client=kms)

    data_key = provider.generate_data_key()
    assert data_key.key_id.startswith("arn:aws:kms:")  # the resolved key, not the alias
    assert provider.unwrap_data_key(data_key.wrapped, data_key.key_id) == data_key.plaintext

    generate, decrypt = kms.calls
    assert generate[1] == {
        "KeyId": "alias/cognee-files",
        "KeySpec": "AES_256",
        "EncryptionContext": KMS_ENCRYPTION_CONTEXT,
    }
    assert decrypt[1]["KeyId"] == data_key.key_id
    assert decrypt[1]["EncryptionContext"] == KMS_ENCRYPTION_CONTEXT


def test_kms_provider_lets_client_errors_through_unchanged():
    """A refused Decrypt is AWS's error, not a corrupt file: it must not be
    re-labelled as a FileDecryptionError."""

    class Failing(FakeKms):
        def decrypt(self, **kwargs):
            raise RuntimeError("AccessDeniedException")

    provider = AwsKmsKeyProvider("alias/x", client=Failing())
    with pytest.raises(RuntimeError, match="AccessDeniedException"):
        provider.unwrap_data_key(b"blob", "arn")


def test_kms_provider_without_boto3_is_a_configuration_error(monkeypatch):
    """Unwrapping on a host without the aws extra names the missing package and
    its remediation instead of reporting a decryption failure."""
    import builtins

    real_import = builtins.__import__

    def no_boto3(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("No module named 'boto3'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_boto3)
    provider = AwsKmsKeyProvider("alias/x")
    with pytest.raises(FileEncryptionConfigurationError, match="boto3") as info:
        provider.unwrap_data_key(b"blob", "arn")
    assert "cognee" in (info.value.remediation or "")


def test_kms_provider_requires_key_id():
    with pytest.raises(FileEncryptionConfigurationError, match="KMS_KEY_ID"):
        AwsKmsKeyProvider("")


def test_unwrap_cache_avoids_repeat_kms_calls():
    reset_key_providers()
    kms = FakeKms()
    provider = AwsKmsKeyProvider("alias/x", client=kms)
    data_key = provider.generate_data_key()

    first = unwrap_data_key_cached(provider, data_key.wrapped, data_key.key_id)
    second = unwrap_data_key_cached(provider, data_key.wrapped, data_key.key_id)

    assert first == second == data_key.plaintext
    assert sum(1 for name, _ in kms.calls if name == "decrypt") == 1
    reset_key_providers()


def test_registry_builds_configured_provider_once(encryption_env):
    first = get_key_provider()
    assert isinstance(first, LocalKeyProvider)
    assert get_key_provider() is first


def test_registry_unknown_provider_lists_options(encryption_env):
    config = encryption_env(provider="vault")
    with pytest.raises(FileEncryptionConfigurationError, match="Unknown FILE_ENCRYPTION_PROVIDER"):
        get_key_provider(config)


def test_custom_provider_can_be_registered(encryption_env):
    class Static:
        name = "static"

        def generate_data_key(self):
            raise NotImplementedError

        def unwrap_data_key(self, wrapped, key_id):
            raise NotImplementedError

    register_file_key_provider("static", lambda config: Static())
    config = encryption_env(provider="static")

    assert isinstance(get_key_provider(config), Static)
