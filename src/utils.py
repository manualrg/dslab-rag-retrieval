from typing import List
import pandas as pd


def corpus_stats(encoding, corpus: List[str]) -> pd.Series:
    lst_docs = [encoding.encode(doc) for doc in corpus]
    data = [len(x) for x in lst_docs]
    return pd.Series(data)