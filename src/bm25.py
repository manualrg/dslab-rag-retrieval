# pip install numpy scipy scikit-learn nltk
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import download

# Asegura recursos NLTK (ejecuta una vez)
download("stopwords", quiet=True)

# Stopwords y stemmer en español
SPANISH_STOPWORDS = set(stopwords.words("spanish"))
STEMMER = SnowballStemmer("spanish")

# Tokenizador: palabras (incluye acentos/ñ), lower, filtra stopwords, aplica stemming
def spanish_stem_tokenizer(text: str):
    tokens = re.findall(r"\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b", text.lower())
    return [STEMMER.stem(t) for t in tokens if t not in SPANISH_STOPWORDS]


class BM25Vectorizer:
    """
    BM25 disperso (CSR) con tokenización/stemming en español.
    - fit(corpus): aprende vocabulario e IDF
    - transform(corpus): devuelve matriz BM25 docs×terms (CSR, float32)
    - transform_queries(queries, binary=True): queries×terms (CSR)
    - score(D, Q): matriz de scores (queries × docs)
    """
    def __init__(self, k1: float = 1.2, b: float = 0.75,
                 ngram_range=(1,1), min_df=1, max_df=1.0):
        self.k1 = float(k1)
        self.b = float(b)
        self.vectorizer = CountVectorizer(
            tokenizer=spanish_stem_tokenizer,
            ngram_range=ngram_range,
            lowercase=True,
            min_df=min_df,
            max_df=max_df
        )
        self.idf_ = None
        self.avgdl_ = None

    def fit(self, corpus):
        # Conteos crudos (docs × terms)
        X = self.vectorizer.fit_transform(corpus).tocsr()
        N, _ = X.shape

        # Longitudes y promedio
        doclen = np.asarray(X.sum(axis=1)).ravel()
        avgdl = float(doclen.mean()) if N else 0.0

        # Frecuencia de documento por término
        df = np.asarray((X > 0).sum(axis=0)).ravel()

        # IDF de BM25 (Robertson/Sparck Jones) con suavizado 0.5
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

        self.idf_ = idf
        self.avgdl_ = np.float32(max(avgdl, 1e-9))
        return self

    def transform(self, corpus) -> sparse.csr_matrix:
        if self.idf_ is None:
            raise RuntimeError("Call fit() first.")

        k1 = self.k1
        b = self.b

        # Matriz de conteos alineada al vocabulario aprendido
        X = self.vectorizer.transform(corpus).tocsr().astype(np.float32, copy=False)

        # Longitud de cada documento del lote
        doclen = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)

        # Norma por documento: k1 * (1 - b + b * |d| / avgdl)
        norms = k1 * (1.0 - b + b * (doclen / self.avgdl_))  # (n_docs,)

        # Expandir norma a cada nnz usando la estructura CSR
        row_counts = np.diff(X.indptr)
        expanded_norms = np.repeat(norms, row_counts).astype(np.float32, copy=False)

        # Accesos directos CSR
        data = X.data      # tf
        cols = X.indices   # término
        idf = self.idf_

        # BM25 vectorizado: idf[j] * ((tf*(k1+1)) / (tf + norm_doc))
        k1p1 = np.float32(k1 + 1.0)
        tf = data
        data[:] = (tf * k1p1) / (tf + expanded_norms)
        data *= idf[cols]

        return X  # CSR docs×terms (float32)

    def transform_queries(self, queries, binary: bool = True) -> sparse.csr_matrix:
        Q = self.vectorizer.transform(queries).tocsr().astype(np.float32, copy=False)
        if binary:
            # Presencia/ausencia clásica de BM25 (qtf = 1)
            Q.data[:] = 1.0
        return Q

    def score(self, doc_bm25: sparse.csr_matrix, query_matrix: sparse.csr_matrix) -> np.ndarray:
        # (docs × terms) @ (terms × queries) -> (docs × queries)
        scores = doc_bm25 @ query_matrix.T
        return scores.T.A  # (queries × docs) denso

    def vocabulary(self):
        """Devuelve dict token->índice (stems)"""
        return self.vectorizer.vocabulary_


# -------------------------
# Ejemplo de uso
# -------------------------
if __name__ == "__main__":
    docs = [
        "El zorro marrón rápido salta sobre el perro perezoso.",
        "Nunca saltes sobre un perro perezoso rápidamente.",
        "Los perros marrones no son zorros."
    ]
    queries = ["zorro marrón", "perro perezoso"]

    bm25 = BM25Vectorizer(k1=1.2, b=0.75).fit(docs)
    D = bm25.transform(docs)              # matriz BM25 (CSR)
    Q = bm25.transform_queries(queries)   # matriz de consultas (CSR)
    scores = bm25.score(D, Q)             # (len(queries) × len(docs))

    for qi, q in enumerate(queries):
        ranking = np.argsort(-scores[qi])
        print(f"\nConsulta: {q!r}")
        for rank, di in enumerate(ranking, 1):
            print(f"  {rank}. doc#{di}  score={scores[qi, di]:.4f}  | {docs[di]}")
