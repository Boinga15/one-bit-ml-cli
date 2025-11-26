from src.llm.parts import *

from src.data.data import load_dataset
import sentencepiece as spm
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    dataset = load_dataset()
    tokenizer = spm.SentencePieceProcessor(model_file = "src/tokenization/tokenizer.model")
    vocabSize = tokenizer.GetPieceSize()

    tokens = np.array(tokenizer.encode(dataset[0]["text"], out_type=int), dtype=np.int32)
    tokens = np.expand_dims(tokens, axis = 0)

    tokensTf = tf.convert_to_tensor(tokens, dtype=tf.int32)

    embedding = PositionalEmbedding(vocabSize, 12)
    result = embedding(tokensTf)
    print(result)

    addAndNormalize = AddAndNormalize()
    result = addAndNormalize([result, result])
    print(result)

    ffn = FFN(768, 3072)
    result = ffn(result)
    print(result)