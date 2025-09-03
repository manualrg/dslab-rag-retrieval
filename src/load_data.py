import uuid
import pandas as pd
from datasets import load_dataset

def ragquas()-> pd.DataFrame:
    return load_dataset("IIC/RagQuAS")['test'].to_pandas()


def prepare_ragquas(df_ragquas) -> pd.DataFrame:
    df_docs = pd.wide_to_long(
        (df_ragquas
        .reset_index()  # crea columna index
        .loc[df_ragquas['variant'] == "question_1"]
        ),
        stubnames=["context_", "text_", "link_"],  # columas a pivotar, solo prefijo
        i=["index", "topic", "variant", "question", "answer"],  # index en el df resultante
        j="j",  # numerador de _1 a _5  (numero de sufijos),
        suffix="\d+"  # forma del sufijo: numero
    )


    print(f"Raw shape: {df_docs.shape}")

    # Como hay text con "", hay que eliminarlos para no indexar algo vacio
    df_docs["text_"] = (df_docs["text_"]
            .apply(str.rstrip)
            .replace({"": pd.NA})
            )

    df_docs = (df_docs
            .dropna()
            .reset_index()
            )

    # agregar un id a cada text_ para poder hacer la evaluacion posteiormente
    df_docs['uuid'] = [str(uuid.uuid4()) for _ in range(len(df_docs))]

    return df_docs