from typing import Type
from uuid import uuid4

import spacy
import textacy
from pydantic import BaseModel

from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.shared.data_models import Edge, KnowledgeGraph, Node, SummarizedContent


class DummyLLMAdapter(LLMInterface):
    nlp = spacy.load("en_core_web_sm")

    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        if str(response_model) == "<class 'cognee.shared.data_models.SummarizedContent'>":
            return dummy_summarize_content(text_input)
        elif str(response_model) == "<class 'cognee.shared.data_models.KnowledgeGraph'>":
            return dummy_extract_knowledge_graph(text_input, self.nlp)
        else:
            raise Exception(
                "Currently dummy acreate_structured_input is only implemented for SummarizedContent and KnowledgeGraph"
            )


def dummy_extract_knowledge_graph(text, nlp):
    doc = nlp(text)
    triples = list(textacy.extract.subject_verb_object_triples(doc))

    nodes = {}
    edges = []
    for triple in triples:
        source = "_".join([str(e) for e in triple.subject])
        target = "_".join([str(e) for e in triple.object])
        nodes[source] = nodes.get(
            source, Node(id=str(uuid4()), name=source, type="object", description="")
        )
        nodes[target] = nodes.get(
            target, Node(id=str(uuid4()), name=target, type="object", description="")
        )
        edge_type = "_".join([str(e) for e in triple.verb])
        edges.append(
            Edge(
                source_node_id=nodes[source].id,
                target_node_id=nodes[target].id,
                relationship_name=edge_type,
            )
        )
    return KnowledgeGraph(nodes=list(nodes.values()), edges=edges)


def dummy_summarize_content(text):
    words = [(word, len(word)) for word in set(text.split(" "))]
    words = sorted(words, key=lambda x: x[1], reverse=True)
    summary = " ".join([word for word, _ in words[:50]])
    description = " ".join([word for word, _ in words[:10]])
    return SummarizedContent(summary=summary, description=description)
