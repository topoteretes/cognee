
import json
import hashlib
import datetime
from pathlib import Path
from typing import Optional, Dict


class CassetteManager:
    def __init__(self, cassette_dir: str = None):
        if cassette_dir is None:
            self.cassette_dir = Path(__file__).parent / "cassettes"
        else:
            self.cassette_dir = Path(cassette_dir)

        self.cassette_dir.mkdir(parents=True, exist_ok=True)
        self._chat_cache: Dict[str, dict] = {}

    def hash_prompt(self, messages: list) -> str:
        return hashlib.sha256(json.dumps(messages).encode()).hexdigest()

    def load_cassette(self, prompt_hash: str) -> Optional[dict]:
        """Load cached chat response from cassette file"""
        cassette_file = self.cassette_dir / f"{prompt_hash}.json"
        if not cassette_file.exists():
            return None

        try:
            with open(cassette_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._chat_cache[prompt_hash] = data["response"]
            return data["response"]
        except Exception as e:
            print(f"Error loading cassette {prompt_hash}: {e}")
            return None

    def replay_chat(self, prompt_hash: str) -> Optional[dict]:
        """Return cached chat response if exists"""
        if prompt_hash in self._chat_cache:
            return self._chat_cache[prompt_hash]

        return self.load_cassette(prompt_hash)

    def record_chat(self, prompt_hash: str, response: dict):
        """Save chat response to cassette file"""
        cassette_file = self.cassette_dir / f"{prompt_hash}.json"

        data = {
            "prompt_hash": prompt_hash,
            "response": response,
            "created": datetime.datetime.now().isoformat(),
        }

        with open(cassette_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def is_missing(self, prompt_hash: str) -> bool:
        return not (self.cassette_dir / f"{prompt_hash}.json").exists()


cassette_manager = CassetteManager()