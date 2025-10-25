import os
from typing import List, Dict, Any
from dataclasses import dataclass
import math

@dataclass
class Chunk:
    id: str
    content: str
    document_id: str
    metadata: Dict[str, Any]

@dataclass
class Association:
    source_chunk_id: str
    target_chunk_id: str
    weight: float
    association_type: str = "semantic_similarity"

class SimpleSimilarity:
    @staticmethod
    def calculate_similarity(text1, text2):
        """Simple cosine similarity without external dependencies"""
        # Convert texts to word sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity (simple alternative)
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        return intersection / union

class LLMClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
    
    def complete(self, prompt: str) -> str:
        """Mock LLM - replace with actual API call if needed"""
        # Simple rule-based mock
        prompt_lower = prompt.lower()
        
        # If both chunks mention similar topics, associate
        if any(topic in prompt_lower for topic in ['river', 'dolphin', 'freshwater']):
            if sum(1 for topic in ['river', 'dolphin', 'freshwater'] if topic in prompt_lower) >= 2:
                return "ASSOCIATE"
        
        return "NO_ASSOCIATION"

class ChunkAssociationTask:
    def __init__(self, llm_client: LLMClient = None, similarity_threshold: float = 0.5):
        self.llm_client = llm_client or LLMClient()
        self.similarity_threshold = similarity_threshold
        self.similarity_calc = SimpleSimilarity()
    
    def process_batch(self, chunks: List[Chunk]) -> List[Association]:
        associations = []
        
        print(f"Processing {len(chunks)} chunks for associations...")
        
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                if chunk1.document_id != chunk2.document_id:
                    association = self.evaluate_association(chunk1, chunk2)
                    if association:
                        associations.append(association)
                        print(f"✅ Association: {chunk1.id} -> {chunk2.id} (weight: {association.weight:.3f})")
        
        print(f"🎯 Total associations created: {len(associations)}")
        return associations
    
    def evaluate_association(self, chunk1: Chunk, chunk2: Chunk) -> Association:
        # Calculate similarity
        similarity = self.similarity_calc.calculate_similarity(chunk1.content, chunk2.content)
        
        if similarity < 0.2:  # Quick filter
            return None
        
        # LLM classification
        prompt = f"""
        Chunk 1: {chunk1.content[:200]}
        Chunk 2: {chunk2.content[:200]}
        Should these be associated? Respond ASSOCIATE or NO_ASSOCIATION:
        """
        
        response = self.llm_client.complete(prompt)
        
        if "ASSOCIATE" in response.upper():
            return Association(
                source_chunk_id=chunk1.id,
                target_chunk_id=chunk2.id,
                weight=similarity
            )
        
        return None

# Test with your example
def main():
    chunks = [
        Chunk("book1_chunk1", "River dolphins are freshwater mammals found in major rivers", "book1", {}),
        Chunk("book2_chapter1", "Freshwater dolphins inhabit rivers like Amazon and use echolocation", "book2", {}),
        Chunk("book3_python", "Python programming for web development", "book3", {})
    ]
    
    task = ChunkAssociationTask()
    associations = task.process_batch(chunks)
    
    print("\n📊 FINAL ASSOCIATIONS:")
    for assoc in associations:
        print(f"   {assoc.source_chunk_id} --[{assoc.weight:.3f}]--> {assoc.target_chunk_id}")

if __name__ == "__main__":
    main()