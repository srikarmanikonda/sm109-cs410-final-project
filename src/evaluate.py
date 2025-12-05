import pandas as pd
import numpy as np
from search_engine import ClinicalTrialsSearch
import os

QUERIES = [
    "lung cancer phase 1",
    "diabetes recruiting",
    "asthma",
    "breast cancer phase 3",
    "pain management",
    "obesity diet",
    "melanoma immunotherapy",
    "hypertension",
    "covid-19 vaccine",
    "depression"
]

def generate_judgment_template(engine, output_path="data/evaluation_judgments.csv"):
    """
    Runs queries and generates a CSV template for manual relevance judgments.
    Includes top 5 results from both baseline and system (deduplicated per query).
    """
    print("Generating judgment template...")
    rows = []

    for q_id, query in enumerate(QUERIES):
        res_base = engine.search(query, top_k=5, use_filters=False)
        res_sys = engine.search(query, top_k=5, use_filters=True)

        docs = {}

        for rank, (idx, row) in enumerate(res_base.iterrows()):
            docs[row['NCTId']] = {
                'query_id': q_id,
                'query': query,
                'doc_id': row['NCTId'],
                'title': row['BriefTitle'],
                'phase': row['Phase'],
                'status': row['OverallStatus']
            }

        for rank, (idx, row) in enumerate(res_sys.iterrows()):
            docs[row['NCTId']] = {
                'query_id': q_id,
                'query': query,
                'doc_id': row['NCTId'],
                'title': row['BriefTitle'],
                'phase': row['Phase'],
                'status': row['OverallStatus']
            }

        for doc in docs.values():
            doc['relevance'] = ''
            rows.append(doc)

    df_judgments = pd.DataFrame(rows)
    df_judgments = df_judgments.sort_values('query_id')

    if not os.path.exists(output_path):
        df_judgments.to_csv(output_path, index=False)
        print(f"Judgment template saved to {output_path}. Please fill in the 'relevance' column (1 for relevant, 0 for not).")
    else:
        print(f"File {output_path} already exists. Skipping generation.")

def compute_metrics(judgment_path="data/evaluation_judgments.csv"):
    """
    Computes Precision@5 and nDCG@5 for Baseline vs System using the judgment file.
    """
    try:
        judgments = pd.read_csv(judgment_path)
        if judgments['relevance'].isnull().all():
            print("Relevance judgments are missing. Please fill the CSV first.")
            return

        judgments['relevance'] = pd.to_numeric(judgments['relevance'], errors='coerce').fillna(0)

        rel_map = dict(zip(zip(judgments['query'], judgments['doc_id']), judgments['relevance']))

        engine = ClinicalTrialsSearch("data/sample_trials.csv")

        metrics = {
            'Baseline': {'p5': [], 'ndcg5': []},
            'System': {'p5': [], 'ndcg5': []}
        }

        for query in QUERIES:


            for system_name, use_filters in [('Baseline', False), ('System', True)]:
                results = engine.search(query, top_k=5, use_filters=use_filters)

                # Calculate P@5
                rel_scores = [rel_map.get((query, row['NCTId']), 0) for _, row in results.iterrows()]
                p5 = sum(rel_scores) / 5.0

                # Calculate nDCG@5
                dcg = sum([(2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel_scores)])

                # IDCG
                ideal_rels = sorted([rel_map.get((query, doc_id), 0) for (q, doc_id) in rel_map if q == query], reverse=True)
                ideal_rels = ideal_rels[:5]
                idcg = sum([(2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_rels)])

                ndcg5 = dcg / idcg if idcg > 0 else 0.0

                metrics[system_name]['p5'].append(p5)
                metrics[system_name]['ndcg5'].append(ndcg5)

        print("\nEvaluation Results:")
        print("-" * 30)
        print(f"{'Metric':<10} | {'Baseline':<10} | {'System':<10}")
        print("-" * 30)
        print(f"{'Mean P@5':<10} | {np.mean(metrics['Baseline']['p5']):.4f}     | {np.mean(metrics['System']['p5']):.4f}")
        print(f"{'Mean nDCG@5':<10} | {np.mean(metrics['Baseline']['ndcg5']):.4f}     | {np.mean(metrics['System']['ndcg5']):.4f}")

    except Exception as e:
        print(f"Error computing metrics: {e}")

if __name__ == "__main__":
    engine = ClinicalTrialsSearch("data/sample_trials.csv")
    generate_judgment_template(engine)




