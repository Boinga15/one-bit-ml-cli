from src.llm.train import fit_model
from src.llm.parts import *

from src.data.data import load_dataset
import sentencepiece as spm
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    tokenizer = spm.SentencePieceProcessor(model_file = "src/tokenization/tokenizer.model")
    vocabSize = tokenizer.GetPieceSize()

    sequenceLength = 512

    model = LLM(vocabSize, 256, 4, 1024, 4, 512)
    fit_model(model, sequenceLength = sequenceLength)