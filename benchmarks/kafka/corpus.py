import random
from dataclasses import dataclass
from typing import Iterator, List

@dataclass
class CorpusDocument:
    id: str
    content: str
    metadata: dict

class CorpusGenerator:
    SEED = 42

    VOCABULARY = [
        "commit", "push", "pull", "request", "merge", "branch", "rebase",
        "refactor", "bug", "fix", "issue", "resolve", "close", "open",
        "architecture", "decision", "api", "endpoint", "database", "migration",
        "index", "cache", "latency", "throughput", "scaling", "deployment",
        "kubernetes", "docker", "pipeline", "ci", "cd", "authentication",
        "authorization", "oauth", "security", "vulnerability", "patch",
        "release", "version", "update", "dependency", "module", "interface",
        "implementation", "class", "method", "function", "variable", "memory",
        "leak", "optimization", "bottleneck", "profiling", "tracing",
        "service", "microservice", "monolith", "rest", "graphql", "grpc"
    ]
    
    DOC_TYPES = [
        "pr_description", 
        "commit_message", 
        "architecture_decision_record", 
        "issue_comment"
    ]

    def __init__(self, num_documents: int, avg_doc_size_bytes: int = 512):
        self.num_documents = num_documents
        self.avg_doc_size_bytes = avg_doc_size_bytes

    def _generate_synthetic_content(self, rand: random.Random) -> str:
        # Generate string of approximately avg_doc_size_bytes
        # Since average word length here is ~7 chars + 1 space = 8 bytes
        # we can estimate target word count
        target_size = rand.randint(int(self.avg_doc_size_bytes * 0.8), int(self.avg_doc_size_bytes * 1.2))
        
        words = []
        current_size = 0
        
        while current_size < target_size:
            word = rand.choice(self.VOCABULARY)
            words.append(word)
            current_size += len(word) + 1
            
        return " ".join(words)

    def generate_stream(self) -> Iterator[CorpusDocument]:
        # Local random instance ensures determinism regardless of global random state
        rand = random.Random(self.SEED)
        
        for index in range(self.num_documents):
            doc_id = f"doc_{index:06d}"
            content = self._generate_synthetic_content(rand)
            
            # Deterministic base timestamp for the event (e.g., commit time)
            # Starts at 2024-01-01 00:00:00 UTC and increments by 1 min per doc
            timestamp_sec = 1704067200 + index * 60
            
            metadata = {
                "type": rand.choice(self.DOC_TYPES),
                "timestamp": timestamp_sec,
                "size_bytes": len(content.encode('utf-8')),
                "index": index
            }
            
            yield CorpusDocument(id=doc_id, content=content, metadata=metadata)

    def generate(self) -> List[CorpusDocument]:
        return list(self.generate_stream())
