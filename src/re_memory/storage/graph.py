"""FalkorDB graph store client.

Stores and queries the semantic knowledge graph (Layer 3).
Uses Cypher query language via FalkorDB's Redis-compatible protocol.
"""

from __future__ import annotations

from datetime import datetime, timezone

from falkordb import FalkorDB as FalkorDBClient


GRAPH_NAME = "re_memory"


class GraphStore:
    """FalkorDB-backed knowledge graph for semantic memory."""

    def __init__(self, host: str = "localhost", port: int = 6379):
        self.host = host
        self.port = port
        self._client: FalkorDBClient | None = None
        self._graph = None

    @property
    def client(self) -> FalkorDBClient:
        if self._client is None:
            self._client = FalkorDBClient(host=self.host, port=self.port)
        return self._client

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.client.select_graph(GRAPH_NAME)
        return self._graph

    def init(self):
        """Create graph indexes."""
        # Create indexes for efficient lookups
        try:
            self.graph.query("CREATE INDEX FOR (e:Entity) ON (e.name)")
        except Exception:
            pass  # Index may already exist
        try:
            self.graph.query("CREATE INDEX FOR (e:Entity) ON (e.type)")
        except Exception:
            pass

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source_event_id: str = "",
        valid_from: str | None = None,
        valid_to: str | None = None,
    ):
        """Add a subject-predicate-object triple to the knowledge graph.

        Bi-temporal model: valid_from/valid_to track when the fact is true,
        ingested_at tracks when we learned it.
        """
        now = datetime.now(timezone.utc).isoformat()
        valid_from = valid_from or now
        ingested_at = now

        query = """
            MERGE (s:Entity {name: $subject})
            MERGE (o:Entity {name: $object})
            CREATE (s)-[r:RELATION {
                predicate: $predicate,
                confidence: $confidence,
                source_event_id: $source_event_id,
                valid_from: $valid_from,
                valid_to: $valid_to,
                ingested_at: $ingested_at,
                active: true
            }]->(o)
            RETURN r
        """
        self.graph.query(
            query,
            params={
                "subject": subject,
                "object": obj,
                "predicate": predicate,
                "confidence": confidence,
                "source_event_id": source_event_id,
                "valid_from": valid_from,
                "valid_to": valid_to or "",
                "ingested_at": ingested_at,
            },
        )

    def query_entity(self, entity_name: str, limit: int = 20) -> list[dict]:
        """Get all relations for an entity (as subject or object)."""
        query = """
            MATCH (s:Entity)-[r:RELATION]->(o:Entity)
            WHERE (s.name = $name OR o.name = $name) AND r.active = true
            RETURN s.name AS subject, r.predicate AS predicate,
                   o.name AS object, r.confidence AS confidence,
                   r.valid_from AS valid_from, r.valid_to AS valid_to
            ORDER BY r.confidence DESC
            LIMIT $limit
        """
        result = self.graph.query(
            query,
            params={"name": entity_name, "limit": limit},
        )
        return [
            {
                "subject": row[0],
                "predicate": row[1],
                "object": row[2],
                "confidence": row[3],
                "valid_from": row[4],
                "valid_to": row[5],
            }
            for row in result.result_set
        ]

    def search_triples(self, text: str, limit: int = 20) -> list[dict]:
        """Search for triples containing any keyword from text.

        Tokenizes the query into individual words, filters stop words,
        and matches triples where any keyword appears in subject, predicate,
        or object names.
        """
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "do", "does", "did", "will", "would", "could", "should", "may",
            "might", "can", "shall", "has", "have", "had", "of", "in", "to",
            "for", "on", "at", "by", "with", "from", "about", "into", "and",
            "or", "but", "not", "no", "so", "if", "then", "than", "that",
            "this", "it", "its", "my", "your", "his", "her", "our", "their",
            "what", "which", "who", "whom", "how", "when", "where", "why",
            "me", "him", "us", "them", "i", "you", "he", "she", "we", "they",
            "tell", "know", "think", "get", "make",
        }
        keywords = [
            w for w in text.lower().split()
            if w not in stop_words and len(w) > 1
        ]
        if not keywords:
            keywords = text.lower().split()

        # Build a WHERE clause that matches any keyword in subject, predicate, or object
        conditions = []
        params: dict = {"limit": limit}
        for i, kw in enumerate(keywords):
            param = f"kw{i}"
            params[param] = kw
            conditions.append(
                f"(toLower(s.name) CONTAINS ${param} OR "
                f"toLower(o.name) CONTAINS ${param} OR "
                f"toLower(r.predicate) CONTAINS ${param})"
            )

        where_clause = f"({' OR '.join(conditions)}) AND r.active = true"

        query = f"""
            MATCH (s:Entity)-[r:RELATION]->(o:Entity)
            WHERE {where_clause}
            RETURN DISTINCT s.name AS subject, r.predicate AS predicate,
                   o.name AS object, r.confidence AS confidence
            ORDER BY r.confidence DESC
            LIMIT $limit
        """
        result = self.graph.query(query, params=params)
        return [
            {
                "subject": row[0],
                "predicate": row[1],
                "object": row[2],
                "confidence": row[3],
            }
            for row in result.result_set
        ]

    def archive_triple(self, subject: str, predicate: str, obj: str):
        """Archive a triple (set active=false, record valid_to)."""
        now = datetime.now(timezone.utc).isoformat()
        self.graph.query(
            """
            MATCH (s:Entity {name: $subject})-[r:RELATION {predicate: $predicate}]->(o:Entity {name: $object})
            WHERE r.active = true
            SET r.active = false, r.valid_to = $now
            """,
            params={
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "now": now,
            },
        )

    def delete_by_source_event(self, source_event_id: str) -> int:
        """Delete all triples that originated from a specific episodic event.

        Returns the number of relationships deleted.
        """
        result = self.graph.query(
            """
            MATCH (s:Entity)-[r:RELATION]->(o:Entity)
            WHERE r.source_event_id = $event_id
            DELETE r
            RETURN COUNT(r) AS deleted
            """,
            params={"event_id": source_event_id},
        )
        # Clean up orphan nodes (entities with no remaining relationships)
        self.graph.query(
            """
            MATCH (n:Entity)
            WHERE NOT (n)-[:RELATION]-() AND NOT (n)<-[:RELATION]-()
            DELETE n
            """
        )
        if result.result_set:
            return result.result_set[0][0]
        return 0

    def get_conflicting_triples(self, subject: str, predicate: str) -> list[dict]:
        """Find active triples that might conflict with a new assertion."""
        query = """
            MATCH (s:Entity {name: $subject})-[r:RELATION {predicate: $predicate}]->(o:Entity)
            WHERE r.active = true
            RETURN s.name AS subject, r.predicate AS predicate,
                   o.name AS object, r.confidence AS confidence
        """
        result = self.graph.query(
            query,
            params={"subject": subject, "predicate": predicate},
        )
        return [
            {
                "subject": row[0],
                "predicate": row[1],
                "object": row[2],
                "confidence": row[3],
            }
            for row in result.result_set
        ]

    def status(self) -> dict:
        """Get graph statistics with a live connectivity check."""
        try:
            # Force a fresh connection to verify FalkorDB is reachable
            check_client = FalkorDBClient(host=self.host, port=self.port)
            check_graph = check_client.select_graph(GRAPH_NAME)
            nodes = check_graph.query("MATCH (n) RETURN COUNT(n)").result_set
            edges = check_graph.query("MATCH ()-[r]->() RETURN COUNT(r)").result_set
            return {
                "graph_nodes": nodes[0][0] if nodes else 0,
                "graph_edges": edges[0][0] if edges else 0,
            }
        except Exception:
            return {"graph_nodes": 0, "graph_edges": 0, "status": "unavailable"}

    def close(self):
        """Close the connection."""
        self._graph = None
        self._client = None
