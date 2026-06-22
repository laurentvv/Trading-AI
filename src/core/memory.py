import json
import logging
import sqlite3
from typing import List
from pathlib import Path

# Use a standard scikit-learn approach for lightweight vectorization
# It perfectly matches the requirement for environments with limited connection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)


class FinancialMemory:
    def __init__(self, db_path: str = "data_cache/finacumen_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False
        self._load_and_fit()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT UNIQUE NOT NULL,
                    gold_answer TEXT NOT NULL,
                    findings TEXT NOT NULL,
                    cautions TEXT NOT NULL
                )
            """)
            conn.commit()

    def _load_and_fit(self):
        """Loads existing memories and fits the TF-IDF vectorizer."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id, question FROM memories")
            rows = cursor.fetchall()

        self.memory_ids = [row[0] for row in rows]
        self.memory_docs = [row[1] for row in rows]

        if self.memory_docs:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.memory_docs)
            self._fitted = True
        else:
            self.tfidf_matrix = None
            self._fitted = False

    def add_memory(self, question: str, gold_answer: str, findings: List[str], cautions: List[str]):
        """Adds a new experience memory."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO memories (question, gold_answer, findings, cautions) VALUES (?, ?, ?, ?)",
                    (question, gold_answer, json.dumps(findings), json.dumps(cautions)),
                )
                conn.commit()
            self._load_and_fit()
        except sqlite3.IntegrityError:
            logger.info("Memory already exists.")

    def retrieve(self, current_context: str, tau: float = 0.65, k_max: int = 5) -> str:
        """
        Retrieves the most similar memories based on a cosine similarity filter.
        Returns a formatted XML <Memory_Block> string.
        """
        if not self._fitted:
            return "<Memory_Handling>\n<!-- No memories available -->\n</Memory_Handling>"

        # Transform the current context query
        query_vec = self.vectorizer.transform([current_context])

        # Calculate cosine similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Filter by threshold tau and sort
        valid_indices = np.where(similarities >= tau)[0]
        if len(valid_indices) == 0:
            return "<Memory_Handling>\n<!-- No relevant memories met the similarity threshold -->\n</Memory_Handling>"

        # Sort by highest similarity
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]

        # Take top k_max
        top_indices = sorted_indices[:k_max]
        top_ids = [self.memory_ids[idx] for idx in top_indices]

        # Fetch the full data for these memories
        memories_data = []
        with sqlite3.connect(self.db_path) as conn:
            # We use an explicit parameter list for the query to prevent SQL injection issues
            placeholders = ",".join("?" for _ in top_ids)
            cursor = conn.execute(
                f"SELECT question, gold_answer, findings, cautions FROM memories WHERE id IN ({placeholders})", top_ids
            )

            # Reorder according to similarities order
            rows_dict = {
                row[0]: row
                for row in cursor.execute(
                    f"SELECT id, question, gold_answer, findings, cautions FROM memories WHERE id IN ({placeholders})",
                    top_ids,
                )
            }
            for mid in top_ids:
                row = rows_dict[mid]
                memories_data.append(
                    {
                        "question": row[1],
                        "gold_answer": row[2],
                        "findings": json.loads(row[3]),
                        "cautions": json.loads(row[4]),
                    }
                )

        # Format as XML
        xml_output = ["<Memory_Handling>"]
        for entry in memories_data:
            xml_output.append("  <Entry>")
            xml_output.append(f"    <Question>{entry['question']}</Question>")

            xml_output.append("    <Findings>")
            for finding in entry["findings"]:
                xml_output.append(f"      - {finding}")
            xml_output.append("    </Findings>")

            xml_output.append("    <Cautions>")
            for caution in entry["cautions"]:
                xml_output.append(f"      - {caution}")
            xml_output.append("    </Cautions>")

            xml_output.append("  </Entry>")
        xml_output.append("</Memory_Handling>")

        return "\n".join(xml_output)

    def seed_initial_memories(self):
        """Pre-populates the database to solve the cold-start problem."""
        seeds = [
            {
                "question": "WTI price has dropped 5% after EIA inventories showed a massive unexpected build.",
                "gold_answer": "SELL. Fundamentals overshadow technical support levels in deep supply shocks.",
                "findings": [
                    "EIA inventory shocks override minor moving average supports.",
                    "WTI reacts violently to physical supply data.",
                ],
                "cautions": [
                    "DO NOT buy the dip on WTI when an EIA massive build is reported.",
                    "Check the Brent spread before attempting to fade the move.",
                ],
            },
            {
                "question": "NASDAQ is making new highs, but RSI is at 82. Should I sell?",
                "gold_answer": "HOLD or BUY. In strong tech bull markets, RSI can stay overbought for weeks.",
                "findings": [
                    "NASDAQ can trend in 'overbought' territory indefinitely during strong momentum.",
                    "RSI > 70 is a sign of strength, not an automatic short.",
                ],
                "cautions": [
                    "DO NOT short NASDAQ solely based on RSI being overbought.",
                    "Wait for a bearish divergence or a break of the 20-day MA before shorting.",
                ],
            },
            {
                "question": "WTI is highly volatile. I should average down my losing position.",
                "gold_answer": "HOLD or SELL. Never average down on a volatile asset losing momentum.",
                "findings": ["Risk management is key. WTI trends can extend much further than expected."],
                "cautions": [
                    "DO NOT average down on a losing WTI position during high volatility periods.",
                    "Always cut losses if price stays below your average cost.",
                ],
            },
        ]

        for seed in seeds:
            self.add_memory(seed["question"], seed["gold_answer"], seed["findings"], seed["cautions"])
