use std::{fs::File, io::{Read, Seek, SeekFrom}};
use nalgebra::*;

use crate::one_bit_llm::parts::{FFN, Layer};

pub fn load_tokens() -> Vec<u8> {
    // Commented for testing purposes. The below code block only extracts the first 16 MB of data.
    // let data = fs::read("../../dataset/tokens.bin").expect("Failed to read file.");

    let mut input_file = File::open("../../dataset/tokens.bin").expect("Failed to read file.");
    let mut data: Vec<u8> = vec![0; 16000];
    let _ = input_file.seek(SeekFrom::Start(0));
    let _ = input_file.read_exact(&mut data);

    data
}

pub fn convert_to_usize(input: Vec<u8>) -> Vec<usize> {
    let mut processed_data: Vec<usize> = vec![];

    for chunk in input.chunks_exact(4) {
        processed_data.push((chunk[0] as usize) * 256^3 + (chunk[1] as usize) * 256^2 + (chunk[2] as usize) * 256 + (chunk[3] as usize));
    }

    processed_data
}

pub fn one_hot_encoding(data: Vec<usize>) -> DMatrix<f64> {
    let mut result: DMatrix<f64> = DMatrix::from_element(*data.iter().max().unwrap() + 1, data.len(), 0.0);

    for (index, element) in data.iter().enumerate() {
        result[(*element, index)] = 1.0;
    }

    result
}

pub fn train(mut model: FFN, training_data: Vec<usize>) {
    // Constants that can be edited to vary the training process.
    const EPOCH_COUNT: usize = 100;
    const BATCH_COUNT_PER_EPOCH: usize = 64;
    const LEARNING_RATE: f64 = 0.0002;
    const SEQUENCE_LENGTH: usize = 256;

    let vocabulary_size = *training_data.iter().max().unwrap();

    // Batch processing
    let mut current_batch_index: usize = 0;

    fn generate_batch (current_batch_index: usize, training_data: &Vec<usize>) -> (DMatrix<usize>, usize, usize) {
        let mut target_index = current_batch_index + SEQUENCE_LENGTH;

        if target_index >= training_data.len() {
            target_index -= training_data.len();
        }

        let mut extracted_data: Vec<usize> = vec![0; SEQUENCE_LENGTH];
        
        for i in 0..(SEQUENCE_LENGTH - 1) {
            //print!("{}, {}, {}", target_index, i, SEQUENCE_LENGTH);
            extracted_data[i] = training_data[target_index + i - SEQUENCE_LENGTH];
        }

        let mut encoded_data: DMatrix<usize> = DMatrix::from_element(1, training_data.len() - 1, 0);

        for i in 0..(training_data.len() - 1) {
            encoded_data[(0, i)] = training_data[i];
        }

        (encoded_data, training_data[training_data.len() - 1], target_index)
    }

    // Training algorithm.
    for epoch in 0..EPOCH_COUNT {
        let mut epoch_loss: f64 = 0.0;
        for _ in 0..BATCH_COUNT_PER_EPOCH {
            let batch_info = generate_batch(current_batch_index, &training_data);
            current_batch_index = batch_info.2;

            let model_result: DMatrix<f64> = model.compute(batch_info.0, true);
            
            let mut target: DMatrix<f64> = DMatrix::from_element(1, vocabulary_size, 0.0);
            let mut loss_gradients: DMatrix<f64> = DMatrix::from_element(1, vocabulary_size, 1.0);

            target[(0, batch_info.1)] = 1.0;
            loss_gradients[(0, batch_info.1)] = -1.0;

            let loss: DMatrix<f64> = (target - model_result).abs();
            epoch_loss += loss.sum();

            model.calculate_gradients(loss_gradients);
            model.adjust_parameters(LEARNING_RATE);
        }

        epoch_loss /= BATCH_COUNT_PER_EPOCH as f64;

        print!("Epoch {} complete.", epoch);
        print!("Loss: {}", epoch_loss)
    }
}