from src.data.data import TokenDataset
from src.llm.test import generate_text

from datasets import load_dataset
import numpy as np
import sentencepiece as spm
import time
import tensorflow as tf
import math

def tokenize_dataset(batchSize = 256, batchLogAmount=1000):
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming = True)
    tokenizer = spm.SentencePieceProcessor(model_file = "src/tokenization/tokenizer.model")

    # Log Info
    totalRows = 0
    totalTokens = 0
    startTime = time.time()
    logNumber = 1

    batch = []

    with open("tokens.bin", "wb") as f:
        for row in dataset:
            batch.append(row["text"])

            if len(batch) == batchSize:
                idsBatch = [tokenizer.encode(t, out_type = int) for t in batch]

                for ids in idsBatch:
                    totalTokens += len(ids)
                    np.array(ids, dtype = np.uint32).tofile(f)
                
                totalRows += batchSize
                batch = []

                if totalRows % (batchSize * batchLogAmount) == 0:
                    elapsed = time.time() - startTime
                    gbSpace = (totalTokens * 4) / (1024 ** 3)

                    print(f"Log #{logNumber}:")
                    print(f"Total Rows: {totalRows}")
                    print(f"Total Tokens: {totalTokens}")
                    print(f"GBs Written: {gbSpace:.2f}")
                    print(f"Hours Elapsed: {elapsed / 3600:.2f}\n\n")

                    logNumber += 1

        # Leftover rows
        if batch:
            idsBatch = [tokenizer.encode(t, out_type = int) for t in batch]

            for ids in idsBatch:
                totalTokens += len(ids)
                np.array(ids, dtype = np.uint32).tofile(f)
            
            totalRows += batchSize
    
    # Last Log
    print(f"Tokenization Completed. Final Statistics:")
    print(f"Total Rows: {totalRows}")
    print(f"Total Tokens: {totalTokens}")
    print(f"GBs Written: {gbSpace:.2f}")
    print(f"Hours Elapsed: {elapsed / 3600:.2f}")


def fit_model(model, trainingData = "../dataset/tokens.bin", sequenceLength = 512, batchSize = 8, shuffleBuffer = 10000, epochs = 1000):
    tokens = np.fromfile(trainingData, dtype = np.uint32)

    # Creating the training data.
    dataset = TokenDataset("../dataset/tokens.bin", sequenceLength, batchSize)

    # Implementing a learning rate scheduler
    def scheduler(step, totalSteps, warmupSteps, learningRate = 3e-5, minimumLearningRate = 5e-6, warmupLearningRate = 1e-5):
        if step < warmupSteps: # Warmup period
            return warmupLearningRate + (learningRate - warmupLearningRate) * (step / warmupSteps)
        
        return learningRate + (learningRate - minimumLearningRate) * (1.0 + math.cos(math.pi * (step - warmupSteps) / (totalSteps - warmupSteps))) / 2.0
    
    # Tracking steps.
    globalSteps = 0
    totalSteps = epochs * len(dataset)
    warmupSteps = math.ceil(0.1 * totalSteps)

    tokenizer = spm.SentencePieceProcessor(model_file = "src/tokenization/tokenizer.model")

    # Actual training step.
    lossFunction = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    optimizer = tf.keras.optimizers.AdamW(learning_rate = 3e-5, weight_decay = 0.1, beta_1 = 0.9, beta_2 = 0.95, epsilon = 1e-8)

    @tf.function
    def train_step(epoch, model, x, y):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = lossFunction(y, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss
    
    # Training
    for epoch in range(epochs):
        print(f"======= Epoch {epoch + 1}/{epochs} =======")

        for i in range(len(dataset)):
            xBatch, yBatch = dataset[i]

            # Optimize learning rate per step.
            learningRate = scheduler(globalSteps, totalSteps, warmupSteps)
            optimizer.learning_rate.assign(learningRate)

            loss = train_step(epoch, model, xBatch, yBatch)

            if 0 == 0:
                print(f"Step {globalSteps + 1}: {loss.numpy():.4f}")
            
            globalSteps += 1
        
        # Saving the model.
        model.save_weights(f"checkpoints/model_{epoch + 1}.h5")
        
        print(f"======= Epoch {epoch + 1} Tech Sample =======")
        print(generate_text(model, tokenizer, "Newton's Third Law states that ", 100))