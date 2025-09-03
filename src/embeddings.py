from typing import List, Dict
import pandas as pd
from qdrant_client import models


def create_index_points(
        embeddings: List[float],
        df_docs: pd.DataFrame,
        col_uuid = "uuid",
        col_text_ = "text_",
        col_topic_ = "topic",
        col_index = "index",
        col_j = "j",
        ):
    assert len(embeddings) == len(df_docs), "embeddings and df_docs should have the same length"

    cols_required = set([col_uuid, col_text_, col_topic_, col_index, col_j])
    cols_existing = set(df_docs.columns)
    assert cols_required.issubset(cols_existing), f"Some columns are not present in df_doc. It must have: {cols_required}"


    lst_pts = []

    for idx, row in df_docs.iterrows():
        point = {
            "id": row[col_uuid],  # uuid aleatorio
            "payload": {
                "text": row[col_text_],
                "topic": row[col_topic_],
                "answer_idx": row[col_index],  # index antes de filtrar las variant == question1
                "text_j": row[col_j],
                "index": row[col_index],  
            },
            "vector": embeddings[idx]  # se puede buscar directamente en la lista por indice del DF pivotado
        }
        lst_pts.append(point)
    
    return lst_pts


def convert_to_qdrant_points(points: List[Dict]) -> List[models.PointStruct]:
    lst_qdrant_pts = []

    for point in points:
        qdrant_point = models.PointStruct(**point)
        lst_qdrant_pts.append(qdrant_point)

    return lst_qdrant_pts
    

def check_query_qdrant(
        client_qdrant,
        index_name,
        query,
        retrieve_k: int,
        *args,
        **kwargs
):

    resp = client_qdrant.query_points(
        collection_name=index_name,
        query=query,
        limit=retrieve_k,
        *args,
        **kwargs
    )


 
    for point in resp.points:
        doc_retrieved = point.payload['text']
        print(f"{point.id=}")
        print(f"{point.score=}")
        print(f"Doc: {doc_retrieved[:500]}...")
        print(f"idx: {point.payload['index']}, j: {point.payload['text_j']}")
        print("-"*30)


def truncation(tokenizer, corpus, max_tokens):
    corpus_trunc = []
    lst_lens = []

    for text in corpus:
        text_tkns = tokenizer.encode(text)
        lst_lens.append(len(text_tkns))
        corpus_trunc.append(
            text_tkns[:max_tokens]
        )

    se_stats = pd.Series(lst_lens)
    print(f"Input corpus tokens statistics: {se_stats.describe()}")
    return corpus_trunc


CHUNK_CONTEXT_PROMPT = """
Aquí está el fragmento que queremos situar dentro de todo el documento
<chunk>
{chunk_content}
</chunk>

Por favor, proporciona un contexto breve y conciso para situar este fragmento dentro del documento en general con el fin de mejorar la recuperación de búsqueda del fragmento.

Responde solo con el contexto conciso y nada más.
"""
