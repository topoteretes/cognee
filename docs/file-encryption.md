# File encryption (application layer)

Cognee can seal every object it writes under `DATA_ROOT_DIRECTORY` with its own
encryption, so the files on disk or in the bucket are ciphertext regardless of
what the filesystem, volume or object store does. The data key for every file is
wrapped by a key you control: a key in the process environment, or a key in AWS
KMS that never leaves your account. This is what lets a deployment answer
"bring your own key" at the Cognee layer rather than only at the AWS layer.

**Scope.** File storage only: the ingested originals (uploads, fetched pages,
raw text) and the text the loaders derive from them. Chunk text, summaries and
entity descriptions in the graph and vector stores, the relational database, the
session cache and logs are **not** covered by this layer; they rely on the
backend's encryption at rest as before. Files Cognee did not copy (an `s3://`
object or a local path referenced in place) are not touched either.

## Turning it on

```bash
FILE_ENCRYPTION_ENABLED=true

# Local key provider (default): a base64-encoded 32-byte key.
FILE_ENCRYPTION_KEY="$(python -c 'import os,base64;print(base64.b64encode(os.urandom(32)).decode())')"
```

Or keep the key in a file readable only by the service user:

```bash
FILE_ENCRYPTION_KEY_FILE=/run/secrets/cognee-file-key
```

Or let AWS KMS hold the key (needs the `aws` extra for boto3, and a role that
may call `GenerateDataKey` and `Decrypt` on the key):

```bash
FILE_ENCRYPTION_ENABLED=true
FILE_ENCRYPTION_PROVIDER=aws_kms
FILE_ENCRYPTION_KMS_KEY_ID=alias/cognee-files      # or a key id / ARN
FILE_ENCRYPTION_KMS_REGION=ca-central-1             # defaults to AWS_REGION
```

Every KMS call carries the encryption context `{"cognee:purpose": "file-storage"}`.
Scope a key policy or grant to that context if the key is shared.

The switch is validated on first use: a missing or malformed key fails the first
`remember()` with a `FileEncryptionConfigurationError` that names the fix,
rather than storing anything unsealed.

## What changes, and what does not

- `remember()`, `add()`, `update()` and the loaders store sealed objects.
  Reading them back (`cognify`, `recall` citations, `GET
  /datasets/{id}/data/{id}/raw`, cloud sync) goes through the same storage layer
  and sees plaintext. Content hashes and sizes in the `data` table are of the
  plaintext, exactly as before.
- Files written **before** the switch was turned on stay readable. The reader
  recognises a sealed file by its header and passes anything else through. Such a
  file is sealed the next time it is written (for example when `update()`
  re-stores it). There is no bulk re-encryption command yet; re-adding the
  documents is the way to seal an existing corpus.
- Loaders open their input by path, so during ingestion a sealed original is
  unsealed into a temporary file (created with the process's private temp-file
  permissions, `0600`) for as long as the loader runs, then removed. This is the
  one point at which plaintext exists outside process memory; put the system
  temp directory on an encrypted volume if that matters to you.
- The system root (embedded databases), cache root and logs are never wrapped;
  wrapping them would corrupt the database engines, not protect them.

## Format

```
COGNEEENC\x01 | header length (uint32 BE) | header (JSON) | chunk 0 | chunk 1 | ...
```

The header carries the format version, cipher, chunk size, provider name, key
id and the wrapped data key. Each chunk is `AES-256-GCM(data_key)` over
`FILE_ENCRYPTION_CHUNK_SIZE` plaintext bytes (default 1 MiB) with a nonce made of
a per-file random prefix plus the chunk index. The additional authenticated data
is the header bytes, the chunk index and a final-chunk flag, so swapping a
header, reordering chunks or truncating at a chunk boundary all fail
authentication instead of decrypting to something plausible. The reader is
seekable: a seek decrypts one chunk, not the file.

## Key rotation (local provider)

Keys are a ring addressed by id. Add the new key, make it active, and old files
keep decrypting under the id recorded in their header:

```bash
FILE_ENCRYPTION_KEYS='{"1": "<old base64 key>", "2": "<new base64 key>"}'
FILE_ENCRYPTION_ACTIVE_KEY_ID=2
```

A single `FILE_ENCRYPTION_KEY` is the ring `{"1": key}`. Drop an old id from the
ring only once no file written under it remains (or after re-adding those
documents). With KMS, rotation is KMS's: enable automatic key rotation on the
key, or point the alias at a new key; previously generated data keys remain
decryptable by the key material that wrapped them.

## Other key management services

```python
from cognee.infrastructure.files.encryption import DataKey, register_file_key_provider


class VaultTransitProvider:
    name = "vault"

    def __init__(self, config):
        # config is the FileEncryptionConfig; FILE_ENCRYPTION_* extras ride on it
        ...

    def generate_data_key(self) -> DataKey:
        # DataKey(plaintext=32 bytes, wrapped=bytes, key_id=str)
        ...

    def unwrap_data_key(self, wrapped: bytes, key_id: str) -> bytes: ...


register_file_key_provider("vault", VaultTransitProvider)
```

Then `FILE_ENCRYPTION_PROVIDER=vault`. Register before the first storage call.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `FILE_ENCRYPTION_ENABLED` | `false` | Seal objects under the data root |
| `FILE_ENCRYPTION_PROVIDER` | `local` | `local`, `aws_kms`, or a registered name |
| `FILE_ENCRYPTION_KEY` | | Base64 32-byte key (local) |
| `FILE_ENCRYPTION_KEYS` | | JSON ring `{"id": "base64 key"}` (local) |
| `FILE_ENCRYPTION_ACTIVE_KEY_ID` | `1` | Ring id new files are written under |
| `FILE_ENCRYPTION_KEY_FILE` | | File holding one base64 key (local) |
| `FILE_ENCRYPTION_KMS_KEY_ID` | | KMS key id, ARN or alias (aws_kms) |
| `FILE_ENCRYPTION_KMS_REGION` | `AWS_REGION` | KMS client region |
| `FILE_ENCRYPTION_KMS_ENDPOINT_URL` | | KMS endpoint override (VPC endpoint, local emulator) |
| `FILE_ENCRYPTION_CHUNK_SIZE` | `1048576` | Plaintext bytes per sealed chunk |

Implementation: `cognee/infrastructure/files/encryption/`.
