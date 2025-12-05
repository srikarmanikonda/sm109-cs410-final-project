import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import re

class ClinicalTrialsSearch:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        self.df = self.df.fillna("")
        self.bm25 = None
        self.tokenized_corpus = []
        self._build_index()

    def _preprocess(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def _build_index(self):
        corpus = (
            self.df['BriefTitle'] + " " +
            self.df['BriefSummary'] + " " +
            self.df['Condition']
        ).tolist()

        self.tokenized_corpus = [self._preprocess(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _parse_query_filters(self, query):
        """
        Extracts potential filters (Phase, Status) from query string.
        """
        filters = {
            'Phase': None,
            'OverallStatus': None,
            'Location': None
        }

        phase_match = re.search(r'phase\s*([0-4]|i{1,3}v?|iv)', query, re.IGNORECASE)
        if phase_match:
            filters['Phase'] = phase_match.group(0).lower()

        status_match = re.search(r'(recruiting|active|completed|not recruiting|enrolling)', query, re.IGNORECASE)
        if status_match:
            filters['OverallStatus'] = status_match.group(0).lower()

        return filters

    def search(self, query, top_k=10, use_filters=True):
        tokenized_query = self._preprocess(query)
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        results = self.df.copy()
        results['bm25_score'] = scores
        results['boost_score'] = 0.0

        if use_filters:
            filters = self._parse_query_filters(query)

            PHASE_BOOST = 5.0
            STATUS_BOOST = 5.0
            LOCATION_BOOST = 3.0

            # Phase matching
            if filters['Phase']:
                p_val = filters['Phase'].lower() # e.g. "phase 1"

                p_num = re.search(r'([0-4]|i{1,3}v?|iv)', p_val)
                if p_num:
                    num = p_num.group(0)
                    variants = [num]
                    if num == '1': variants.append('i')
                    if num == '2': variants.append('ii')
                    if num == '3': variants.append('iii')
                    if num == '4': variants.append('iv')
                    if num == 'i': variants.append('1')
                    if num == 'ii': variants.append('2')
                    if num == 'iii': variants.append('3')
                    if num == 'iv': variants.append('4')

                pattern = r'phase\s*(?:' + '|'.join(variants) + r')\b'

                results['boost_score'] += results['Phase'].str.lower().str.contains(pattern, regex=True).astype(float) * PHASE_BOOST
            else:
                results['boost_score'] += results['Phase'].str.lower().str.contains(filters['Phase'], regex=False).astype(float) * PHASE_BOOST

            if filters['OverallStatus']:
                results['boost_score'] += results['OverallStatus'].str.lower().str.contains(filters['OverallStatus'], regex=False).astype(float) * STATUS_BOOST




            q_words = set(tokenized_query)


            def check_location_match(row):
                city = str(row['LocationCity']).lower()
                state = str(row['LocationState']).lower()
                q = query.lower()

                score = 0.0
                if city and city in q:
                    score += LOCATION_BOOST
                if state and state in q:

                    if len(state) > 2 or state in tokenized_query:
                        score += LOCATION_BOOST
                return score

            results['location_boost'] = results.apply(check_location_match, axis=1)
            results['boost_score'] += results['location_boost']

        results['final_score'] = results['bm25_score'] + results['boost_score']

        results = results.sort_values(by='final_score', ascending=False)

        return results.head(top_k)

if __name__ == "__main__":
    engine = ClinicalTrialsSearch("data/sample_trials.csv")
    print("Index built.")

    test_query = "lung cancer recruiting phase 1"
    print(f"Searching for: {test_query}")
    results = engine.search(test_query)
    print(results[['BriefTitle', 'Phase', 'OverallStatus', 'bm25_score', 'boost_score']].head())
