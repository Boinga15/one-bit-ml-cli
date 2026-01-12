use crate::matrix::matrix::Matrix;

pub mod one_bit_llm;
pub mod algorithms;
pub mod matrix;

//use crate::{algorithms::train::train, one_bit_llm::parts::LLM};

fn main() {
    /*
    let data: Vec<u8> = algorithms::train::load_tokens();
    let data_converted: Vec<usize> = algorithms::train::convert_to_usize(data.clone());
    
    let model: LLM = LLM::new(*data_converted.iter().max().unwrap(), 512, 8, 1024, 6, 255);
    train(model, data_converted);
    */

    let mut matrix_1: Matrix<f32> = Matrix::new(5, 3, 3.0);

    matrix_1.set(0, 2, 6.0);
    matrix_1.display();

    let matrix_2: Matrix<f32> = matrix_1.col_softmax();
    matrix_2.display();
}