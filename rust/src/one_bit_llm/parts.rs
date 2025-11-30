use nalgebra::*;


pub trait Layer {
    fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>);
    fn adjust_parameters(&mut self, learning_rate: f64);
}


pub struct Dense {
    pub weights: DMatrix<f64>,
    pub weight_gradients: DMatrix<f64>,

    pub quantized_weights: DMatrix<f64>,
    
    pub biases: DMatrix<f64>,
    pub bias_gradients: DMatrix<f64>,

    prev_input: DMatrix<f64>
}

impl Dense {
    pub fn calculate(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        self.prev_input = input;
        &self.weights * &self.prev_input + &self.biases
    }
}

impl Layer for Dense {
    fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>) {
        self.weight_gradients = &previous_gradient * &self.prev_input.transpose();
        self.bias_gradients = previous_gradient.clone();
    }
    
    fn adjust_parameters(&mut self, learning_rate: f64) {
        self.weights -= &self.weight_gradients * learning_rate;
        self.biases -= &self.bias_gradients * learning_rate;
    }
}