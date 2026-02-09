import os
from typing import List
from cog.torque import Graph
from llama_index.core import PropertyGraphIndex, Document
from llama_index.core.graph_stores import PropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.graph_stores.types import EntityNode, Relation

# 1. The CogDB Bridge
class CogPropertyGraphStore(PropertyGraphStore):
    def __init__(self, graph_name: str):
        # Initializes or connects to a local CogDB instance
        self.graph = Graph(graph_name)

    def upsert_nodes(self, nodes: List[EntityNode]) -> None:
        for node in nodes:
            self.graph.put(node.name, "is_a", node.label)

    def upsert_relations(self, relations: List[Relation]) -> None:
        for rel in relations:
            # Triple: Subject (source) -> Predicate (label) -> Object (target)
            self.graph.put(rel.source_id, rel.label, rel.target_id)

    def get(self, subj: str) -> List[List[str]]:
        # Retrieval logic: finds all facts connected to the subject
        res = self.graph.v(subj).out().tag("edge").all()
        return [[subj, r['edge'], r['id']] for r in res.get('result', [])]

# 2. Public Demo Data
space_lore = """
The planet Zephyr-7 is located in the Andromeda Sector. 
It is inhabited by the Kryon species, who are master architects. 
The Kryons built the Great Stellar Gate using Chronos-Steel. 
Chronos-Steel is a rare alloy harvested from dying stars.
The Great Stellar Gate allows for instantaneous travel to the Milky Way.
"""
documents = [Document(text=space_lore)]

# 3. Execution
# Note: Ensure OPENAI_API_KEY is set in your environment
cog_store = CogPropertyGraphStore("galactic_knowledge_base")

index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=cog_store,
    kg_extractors=[
        SchemaLLMPathExtractor(
            possible_entities=["PLANET", "SPECIES", "TECHNOLOGY", "MATERIAL", "SECTOR"],
            possible_relations=["LOCATED_IN", "INHABITED_BY", "BUILT", "MADE_OF", "ALLOWS_TRAVEL_TO"]
        )
    ]
)

# 4. The "Magic" Query
query_engine = index.as_query_engine()
response = query_engine.query("What material was used to build the gate on Zephyr-7, and what does it allow?")

print(f"\n--- AI RESPONSE ---\n{response}")
