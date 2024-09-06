from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval

from collections import defaultdict
import pandas as pd

class LegalSALContractsQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LegalSALContractsQA",
        description="A custom retrieval task dataset including queries and relevant documents.",
        reference="https://example.com/dataset",  # Replace with a valid URL for your dataset
        dataset={
            "path": "sal-contract-qa-100/",  # Replace with your dataset path
            "revision": "main",    # Use the relevant revision if any
        },
        type="Retrieval",
        category="s2p",  # Assuming sentence-to-paragraph retrieval, adjust if different
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],  # Adjust language as needed
        main_score="ndcg_at_10",
        date=None,
        domains=["Legal"],  # Use a valid domain category
        task_subtypes=["Question answering"],  # Use a valid task subtype
        license="Your dataset license here",
        annotations_creators="derived",
        dialect=None,
        sample_creation="found",
        bibtex_citation="""@misc{yourcitation,
  title={Your Dataset Title},
  author={Your Name},
  year={2024},
  publisher={Your Publisher},
  note={Your Notes Here}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 200,  # Replace with your stats
                    "average_query_length": 50,     # Replace with your stats
                    "num_documents": 1000,          # Replace with your stats
                    "num_queries": 100,             # Replace with your stats
                    "average_relevant_docs_per_query": 1.0,  # Replace with your stats
                }
            },
        },
    )

    def load_data(self, **kwargs):
        # Initialize corpus, queries, and relevant_docs dictionaries
        self.corpus = {}
        self.queries = {}
        self.relevant_docs = {}
        split = 'test'

        # Load data from CSV files
        queries_df = pd.read_csv(f'{self.metadata.dataset["path"]}mteb_queries.csv')
        contexts_df = pd.read_csv(f'{self.metadata.dataset["path"]}mteb_contexts.csv')
        relevance_pairs_df = pd.read_csv(f'{self.metadata.dataset["path"]}mteb_relevance_pairs.csv')

        # Convert data to dictionaries for MTEB format
        self.queries[split] = {row['id']: row['text'] for _, row in queries_df.iterrows()}
        self.corpus[split] = {row['id']: {'text': row['text']} for _, row in contexts_df.iterrows()}
        self.relevant_docs[split] = defaultdict(dict)

        for _, row in relevance_pairs_df.iterrows():
            self.relevant_docs[split][row['query_id']][row['context_id']] = int(row['score'])

        self.data_loaded = True

