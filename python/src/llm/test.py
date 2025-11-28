import tensorflow as tf

def sample_from_logits(logits, temperature = 1.0, topK = 50, topP = 0.95):
    # Temperature
    if temperature > 0:
        logits /= temperature
    
    # K filter
    if topK > 0:
        values, _ = tf.math.top_k(logits, k = topK)
        threshold = values[..., -1, tf.newaxis]
        
        logits = tf.where(logits < threshold, tf.ones_like(logits) * -1e10, logits)
    
    # P filter
    if 0.0 < topP < 1.0:
        sortedLogits = tf.sort(logits, direction = "DESCENDING")
        cumulativeProbs = tf.cumsum(tf.nn.softmax(sortedLogits, axis = -1), axis = -1)

        cutoff = cumulativeProbs > topP
        cutoff = tf.roll(cutoff, shift = 1, axis = -1)
        cutoff = tf.where(tf.equal(tf.range(tf.shape(logits)[-1]), 0), False, cutoff)

        sortedIndices = tf.argsort(logits, direction = "DESCENDING")
        mask = tf.zeros_like(logits, dtype = tf.bool)
        mask = tf.tensor_scatter_nd_update(mask, indices=tf.expand_dims(sortedIndices, -1), updates = cutoff)

        logits = tf.where(mask, tf.ones_like(logits) * -1e10, logits)

    probs = tf.nn.softmax(logits, axis = -1)
    token = tf.random.categorical(tf.math.log(probs), num_samples = 1)
    return int(token[0, 0])


def generate_text(model, tokenizer, prompt, maxNewTokens, temperature = 1.0, topK = 50, topP = 0.95):
    tokens = tokenizer.encode(prompt, out_type = int)
    tokens = tokens[-model.max_positoins:]
    tokens = tf.constant(tokens, dtype = tf.int32)[tf.newaxis, :]

    for _ in range(maxNewTokens):
        logits = model(tokens)

        logitsLast = logits[:, -1, :]
        logitsLast = tf.squeeze(logitsLast, axis = 0)

        nextToken = sample_from_logits(logitsLast, temperature, topK, topP)

        nextToken = tf.constant([[nextToken]], dtype=tf.int32)
        tokens = tf.concat([tokens, nextToken], axis = 1)

        # Early stopping clause
        if nextToken == tokenizer.eos_id():
            break
    
    outputTokens = tokens.numpy().tolist()[0]
    return tokenizer.decode(outputTokens)