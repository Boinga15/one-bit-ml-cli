from datasets import load_from_disk

import tensorflow as tf
import numpy as np

def load_dataset():
    dataset = load_from_disk("../dataset/dataset")
    return dataset

class TokenDataset(tf.keras.utils.Sequence):
    def __init__(self, tokenFile, sequenceLength = 512, batchSize = 8, sequentialStepSize = 64):
        self.tokens = np.memmap(tokenFile, dtype = np.uint32, mode = "r")
        self.sequenceLength = sequenceLength
        self.batchSize = batchSize
        self.numTokens = len(self.tokens)
        self.sequentialStepSize = sequentialStepSize

        self.numSequences = (self.numTokens - 1) // self.sequenceLength

    def __len__(self):
        return self.numSequences // self.batchSize

    def __getitem__(self, idx):
        batchX = []
        batchY = []

        startID = idx * self.batchSize

        for i in range(self.batchSize):
            sequenceStart = (startID + i) * self.sequenceLength
            sequenceEnd = sequenceStart + self.sequenceLength + 1
            
            sequence = self.tokens[sequenceStart : sequenceEnd]

            batchX.append(sequence[:-1])
            batchY.append(sequence[1:])

        return np.array(batchX, dtype = np.int32), np.array(batchY, dtype = np.int32)