use nalgebra::*;
use rand::{self, Rng};
use std::{f64::consts::PI};


// Helper Functions.
fn pow_matrix(matrix: DMatrix<f64>, pow: i32) -> DMatrix<f64> {
    let mut resultant_matrix: DMatrix<f64> = DMatrix::from_element(matrix.shape().0, matrix.shape().1, 0.0);

    for row in 0..matrix.shape().0 {
        for column in 0..matrix.shape().1 {
            resultant_matrix[(row, column)] = matrix[(row, column)].powi(pow);
        }
    }

    resultant_matrix
}

fn tanh_matrix(matrix: DMatrix<f64>) -> DMatrix<f64> {
    let mut resultant_matrix: DMatrix<f64> = DMatrix::from_element(matrix.shape().0, matrix.shape().1, 0.0);

    for row in 0..matrix.shape().0 {
        for column in 0..matrix.shape().1 {
            resultant_matrix[(row, column)] = matrix[(row, column)].tanh();
        }
    }

    resultant_matrix
}

fn sech_matrix(matrix: DMatrix<f64>) -> DMatrix<f64> {
    let mut resultant_matrix: DMatrix<f64> = DMatrix::from_element(matrix.shape().0, matrix.shape().1, 0.0);

    for row in 0..matrix.shape().0 {
        for column in 0..matrix.shape().1 {
            resultant_matrix[(row, column)] = 1.0 / matrix[(row, column)].cosh();
        }
    }

    resultant_matrix
}

fn element_multiplication_matrix(matrix_1: DMatrix<f64>, matrix_2: DMatrix<f64>) -> DMatrix<f64> {
    let mut resultant_matrix: DMatrix<f64> = DMatrix::from_element(matrix_1.shape().0, matrix_1.shape().1, 0.0);

    for row in 0..matrix_1.shape().0 {
        for column in 0..matrix_1.shape().1 {
            resultant_matrix[(row, column)] = matrix_1[(row, column)] * matrix_2[(row, column)];
        }
    }

    resultant_matrix
}

fn softmax_matrix(matrix: DMatrix<f64>) -> DMatrix<f64> {
    let mut resultant_matrix: DMatrix<f64> = matrix.clone();
    
    for col in 0..matrix.shape().1 {
        let mut total: f64 = 0.0;

        for row in 0..matrix.shape().0 {
            total += matrix[(row, col)].exp();
        }

        for row in 0..matrix.shape().0 {
            resultant_matrix[(row, col)] /= total;
        }
    }

    resultant_matrix
}
  


pub trait Layer {
    fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>) -> DMatrix<f64>;
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
    pub fn new(nodes: usize, input_size: usize) -> Self {
        Dense {
            weights: DMatrix::new_random(nodes, input_size),
            weight_gradients: DMatrix::from_element(nodes, input_size, 0.0),
            quantized_weights: DMatrix::from_element(nodes, input_size, 0.0),

            biases: DMatrix::new_random(nodes, 1),
            bias_gradients: DMatrix::from_element(nodes, 1, 0.0),

            prev_input: DMatrix::from_element(nodes, 1, 0.0)
        }
    }

    pub fn calculate(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        self.prev_input = input;
        &self.weights * &self.prev_input + &self.biases
    }
}

impl Layer for Dense {
    fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>) -> DMatrix<f64> {
        self.weight_gradients = &previous_gradient * &self.prev_input.transpose();
        self.bias_gradients = previous_gradient.clone();

        return self.weight_gradients.transpose() * previous_gradient;
    }
    
    fn adjust_parameters(&mut self, learning_rate: f64) {
        self.weights -= &self.weight_gradients * learning_rate;
        self.biases -= &self.bias_gradients * learning_rate;
    }
}


pub struct GELU {}

impl GELU {
    pub fn calculate(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        let inside_result: DMatrix<f64> = (&input + 0.044715 * pow_matrix(input.clone(), 3)) * (2.0 / (PI));
        let one_matrix: DMatrix<f64> = DMatrix::from_element(inside_result.shape().0, inside_result.shape().1, 1.0);

        0.5 * &input * (one_matrix + tanh_matrix(inside_result))
    }
}


pub struct FFN {
    pub d_model: usize,
    pub inner_size: usize,

    pub first_layer: Dense,
    pub second_layer: Dense,
    pub activation_layer: GELU,

    prev_inbetween_output: DMatrix<f64>,
    prev_output_result: DMatrix<f64>
}

impl FFN {
    pub fn new(d_model: usize, inner_size: usize) -> Self {
        FFN {
            d_model: d_model,
            inner_size: inner_size,

            first_layer: Dense::new(inner_size, d_model),
            second_layer: Dense::new(d_model, inner_size),
            activation_layer: GELU {},

            prev_inbetween_output: DMatrix::from_element(inner_size, 1, 0.0),
            prev_output_result: DMatrix::from_element(d_model, 1, 0.0),
        }
    }

    pub fn calculate(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        self.prev_inbetween_output = self.first_layer.calculate(input);
        let activated_output: DMatrix<f64> = self.activation_layer.calculate(self.prev_inbetween_output.clone());
        self.prev_output_result = self.second_layer.calculate(activated_output);

        self.prev_output_result.clone()
    }
}

impl Layer for FFN {
    fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>) -> DMatrix<f64> {
        const FIXED_CONSTANT: f64 = 0.044715;

        let activation_gradients: DMatrix<f64> = self.second_layer.calculate_gradients(previous_gradient.clone());
        let one_matrix: DMatrix<f64> = DMatrix::from_element(activation_gradients.shape().0, activation_gradients.shape().1, 1.0);

        let chained_gradients: DMatrix<f64> = 0.5 * (one_matrix.clone() + tanh_matrix((2.0 / PI) * (activation_gradients.clone() +  FIXED_CONSTANT * pow_matrix(activation_gradients.clone(), 3)))) + 0.5 * activation_gradients.clone() * (((2.0 / PI) * one_matrix.clone() + (((FIXED_CONSTANT * 6.0) / PI) * pow_matrix(activation_gradients.clone(), 2))) * pow_matrix(sech_matrix((2.0 / PI) * one_matrix.clone() + ((2.0 * FIXED_CONSTANT) / PI) * pow_matrix(activation_gradients.clone(), 3)), 2));
        let loss_adjusted_gradients: DMatrix<f64> = element_multiplication_matrix(activation_gradients, chained_gradients);

        let final_gradients: DMatrix<f64> = self.first_layer.calculate_gradients(loss_adjusted_gradients);
        return final_gradients;
    }
    
    fn adjust_parameters(&mut self, learning_rate: f64) {
        self.first_layer.adjust_parameters(learning_rate);
        self.second_layer.adjust_parameters(learning_rate);
    }
}


pub struct LayerNorm {
    pub scaling_parameter: f64,
    pub shifting_parameter: f64,

    scaling_gradient: f64,
    shifting_gradient: f64,

    prev_mean: DMatrix<f64>,
    prev_variance: DMatrix<f64>,

    prev_inputs: DMatrix<f64>,
    prev_normalized_inputs: DMatrix<f64>
}

impl LayerNorm {
    pub fn new(range: f64) -> LayerNorm {
        LayerNorm {
            scaling_parameter: rand::thread_rng().gen_range((-1.0 * range)..range),
            scaling_gradient: 0.0,

            shifting_parameter: rand::thread_rng().gen_range((-1.0 * range)..range),
            shifting_gradient: 0.0,

            prev_mean: DMatrix::from_element(1, 1, 0.0),
            prev_variance: DMatrix::from_element(1, 1, 0.0),

            prev_inputs: DMatrix::from_element(1, 1, 0.0),
            prev_normalized_inputs: DMatrix::from_element(1, 1, 0.0)
        }
    }

    pub fn calculate(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        const EPSILON_CONSTANT: f64 = 0.00001;

        self.prev_inputs = input.clone();

        // Calculating mean and variance.
        let mut means: DMatrix<f64> = DMatrix::from_element(1, input.shape().1, 0.0);
        let mut variances: DMatrix<f64> = DMatrix::from_element(1, input.shape().1, 0.0);
    
        for col in 0..input.shape().1 {
            for row in 0..input.shape().0 {
                means[(0, col)] += input[(row, col)];
            }

            means[(0, col)] /= input.shape().0 as f64;
        }

        self.prev_mean = means.clone();

        for col in 0..input.shape().1 {
            for row in 0..input.shape().0 {
                variances[(0, col)] += (input[(row, col)] - means[(0, col)]).powi(2);
            }

            variances[(0, col)] /= input.shape().0 as f64;
        }

        self.prev_variance = variances.clone();

        // Normalizing Inputs
        let mut normalized: DMatrix<f64> = DMatrix::from_element(input.shape().0, input.shape().1, 0.0);

        for col in 0..input.shape().1 {
            for row in 0..input.shape().0 {
                normalized[(row, col)] = (input[(row, col)] - means[(0, col)]) / (variances[(0, col)] + EPSILON_CONSTANT).sqrt();
            }
        }

        self.prev_normalized_inputs = normalized.clone();

        let ones_matrix: DMatrix<f64> = DMatrix::from_element(input.shape().0, input.shape().1, 1.0);

        self.scaling_parameter * normalized + (ones_matrix * self.shifting_parameter)
    }
}

impl Layer for LayerNorm {
    fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>) -> DMatrix<f64> {
        const EPSILON_CONSTANT: f64 = 0.00001;
        
        for row in 0..previous_gradient.shape().0 {
            for col in 0..previous_gradient.shape().1 {
                self.scaling_gradient += self.prev_normalized_inputs[(row, col)] * previous_gradient[(row, col)];
                self.shifting_gradient += previous_gradient[(row, col)];
            }
        }

        let mut normalize_gradients: DMatrix<f64> = DMatrix::from_element(self.prev_normalized_inputs.shape().0, self.prev_normalized_inputs.shape().1, self.scaling_parameter);
        
        for row in 0..previous_gradient.shape().0 {
            for col in 0..previous_gradient.shape().1 {
                normalize_gradients[(row, col)] *= previous_gradient[(row, col)];
            }
        }
        
        // Calculating return gradients.
        let mut input_gradients: DMatrix<f64> = DMatrix::from_element(self.prev_inputs.shape().0, self.prev_inputs.shape().1, 0.0);
        let mut mean_gradients: DMatrix<f64> = DMatrix::from_element(1, self.prev_inputs.shape().1, 0.0);
        let mut variance_gradients: DMatrix<f64> = DMatrix::from_element(1, self.prev_inputs.shape().1, 0.0);

        for col in 0..variance_gradients.shape().1 {
            for row in 0..normalize_gradients.shape().0 {
                variance_gradients[(0, col)] += normalize_gradients[(row, col)] * (-1.0 * self.prev_variance[(0, col)].sqrt() * (self.prev_inputs[(row, col)] - self.prev_mean[(0, col)]) * (self.prev_variance[(0, col)] + EPSILON_CONSTANT).powf(-1.0 * (3.0 / 2.0)));
            }
        }

        for col in 0..mean_gradients.shape().1 {
            let mut column_total: f64 = 0.0;

            for row in 0..normalize_gradients.shape().0 {
                column_total += self.prev_inputs[(row, col)];
                mean_gradients[(0, col)] += normalize_gradients[(row, col)] * (-1.0 / (self.prev_variance[(0, col)] + EPSILON_CONSTANT).sqrt());
            }

            mean_gradients[(0, col)] += variance_gradients[(0, col)] * ((2.0 * self.prev_mean[(0, col)]) - ((2.0 * column_total) / (normalize_gradients.shape().0 as f64)))
        }

        for col in 0..variance_gradients.shape().1 {
            for row in 0..normalize_gradients.shape().0 {
                input_gradients[(row, col)] += normalize_gradients[(row, col)] * (1.0 / (self.prev_variance[(0, col)] + EPSILON_CONSTANT).sqrt());
                input_gradients[(row, col)] += variance_gradients[(0, col)] * ((2.0 * self.prev_inputs[(row, col)]) - (2.0 * self.prev_mean[(0, col)])) / normalize_gradients.shape().0 as f64;
                input_gradients[(row, col)] += mean_gradients[(0, col)] * (1.0 / normalize_gradients.shape().0 as f64);
            }
        }

        input_gradients
    }
    
    fn adjust_parameters(&mut self, learning_rate: f64) {
        self.scaling_parameter -= self.scaling_gradient * learning_rate;
        self.shifting_parameter -= self.shifting_gradient * learning_rate;
    }
}


pub struct ScaledDotProductAttention {
    prev_q: DMatrix<f64>,
    prev_k: DMatrix<f64>,
    prev_v: DMatrix<f64>,
    prev_mask_result: DMatrix<f64>,
    prev_softmax_result: DMatrix<f64>
}

impl ScaledDotProductAttention {
    pub fn new() -> ScaledDotProductAttention {
        ScaledDotProductAttention {
            prev_q: DMatrix::from_element(1, 1, 0.0),
            prev_k: DMatrix::from_element(1, 1, 0.0),
            prev_v: DMatrix::from_element(1, 1, 0.0),
            prev_mask_result: DMatrix::from_element(1, 1, 0.0),
            prev_softmax_result: DMatrix::from_element(1, 1, 0.0)
        }
    }

    pub fn calculate(&mut self, q: DMatrix<f64>, k: DMatrix<f64>, v: DMatrix<f64>, mask: DMatrix<f64>) -> DMatrix<f64> {
        let scaled_multiplication: DMatrix<f64> = (q * k.transpose()) / (k.shape().1 as f64);
        
        let masked_inputs: DMatrix<f64> = scaled_multiplication + mask;
        self.prev_mask_result = masked_inputs.clone();
        
        let softmax_result: DMatrix<f64> = softmax_matrix(masked_inputs);
        self.prev_softmax_result = softmax_result.clone();
        
        let final_result: DMatrix<f64> = softmax_result * v;

        final_result
    }

    pub fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        let mut exponent_sums: DMatrix<f64> = DMatrix::from_element(1, self.prev_mask_result.shape().1, 0.0);

        for col in 0..exponent_sums.shape().1 {
            for row in 0..self.prev_mask_result.shape().0 {
                exponent_sums[(0, col)] += self.prev_mask_result[(row, col)].exp();
            }
        }

        let v_gradients: DMatrix<f64> = &self.prev_softmax_result * &previous_gradient;

        let mut softmax_gradients: DMatrix<f64> = &previous_gradient * self.prev_v.transpose();

        for col in 0..softmax_gradients.shape().1 {
            for row in 0..softmax_gradients.shape().0 {
                softmax_gradients[(row, col)] *= &self.prev_softmax_result[(row, col)].exp() / exponent_sums[(0, col)] - (2.0 * &self.prev_softmax_result[(row, col)]).exp() / (exponent_sums[(0, col)]).powi(2);
            }
        }

        let k_gradients: DMatrix<f64> = &softmax_gradients * &self.prev_q * (1.0 / (self.prev_k.shape().1 as f64).sqrt());
        let q_gradients: DMatrix<f64> = &softmax_gradients * &self.prev_k * (1.0 / (self.prev_k.shape().1 as f64).sqrt());

        let final_gradients: (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) = (q_gradients, k_gradients, v_gradients);
        
        final_gradients
    }
}


pub struct MultiHeadAttention {
    pub d_model: usize,
    pub number_of_heads: usize,

    pub q_linear: Vec<Dense>,
    pub k_linear: Vec<Dense>,
    pub v_linear: Vec<Dense>,

    pub prev_k: DMatrix<f64>,

    pub scaled_layers: Vec<ScaledDotProductAttention>,

    pub final_linear: Dense
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, number_of_heads: usize) -> MultiHeadAttention {
        let mut q_linear: Vec<Dense> = vec![];
        let mut k_linear: Vec<Dense> = vec![];
        let mut v_linear: Vec<Dense> = vec![];
        let mut scaled_layers: Vec<ScaledDotProductAttention> = vec![];

        for _ in 0..number_of_heads {
            q_linear.push(Dense::new(d_model, d_model));
            k_linear.push(Dense::new(d_model, d_model));
            v_linear.push(Dense::new(d_model, d_model));
            scaled_layers.push(ScaledDotProductAttention::new());
        }

        MultiHeadAttention { 
            d_model: d_model,
            number_of_heads: number_of_heads,
            q_linear: q_linear,
            k_linear: k_linear,
            v_linear: v_linear,
            prev_k: DMatrix::from_element(1, 1, 0.0),
            scaled_layers: scaled_layers,
            final_linear: Dense::new(d_model, d_model * number_of_heads)
        }
    }

    pub fn calculate(&mut self, q: DMatrix<f64>, k: DMatrix<f64>, v: DMatrix<f64>, mask: DMatrix<f64>) -> DMatrix<f64> {
        let mut q_results: Vec<DMatrix<f64>> = vec![];
        let mut k_results: Vec<DMatrix<f64>> = vec![];
        let mut v_results: Vec<DMatrix<f64>> = vec![];

        self.prev_k = k.clone();

        for i in 0..self.number_of_heads {
            q_results.push(self.q_linear[i].calculate(q.clone()));
            k_results.push(self.k_linear[i].calculate(k.clone()));
            v_results.push(self.v_linear[i].calculate(v.clone()));
        }

        let mut scaled_results: Vec<DMatrix<f64>> = vec![];

        for i in 0..self.number_of_heads {
            scaled_results.push(self.scaled_layers[i].calculate(q_results[i].clone(), k_results[i].clone(), v_results[i].clone(), mask.clone()))
        }

        let mut concatenated_result: DMatrix<f64> = DMatrix::from_element(scaled_results[0].shape().0, scaled_results[0].shape().1 * self.number_of_heads, 0.0);
        
        for i in 0..self.number_of_heads {
            for row in 0..scaled_results[i].shape().0 {
                for col in 0..scaled_results[i].shape().1 {
                    concatenated_result[(row, col + (i * scaled_results[0].shape().1))] = scaled_results[i][(row, col)];
                }
            }
        }

        self.final_linear.calculate(concatenated_result)
    }

    pub fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        let final_layer_result: DMatrix<f64> = self.final_linear.calculate_gradients(previous_gradient);
        
        let mut split_results: Vec<DMatrix<f64>> = vec![];

        for i in 0..self.number_of_heads {
            let mut new_result: DMatrix<f64> = DMatrix::from_element(self.d_model, self.d_model, 0.0);

            for row in 0..self.d_model {
                for col in 0..self.d_model {
                    new_result[(row, col)] = final_layer_result[(row, col + (i * self.d_model))];
                }
            }

            split_results.push(new_result);
        }

        let mut concat_gradients: Vec<(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)> = vec![];

        for i in 0..self.number_of_heads {
            concat_gradients.push(self.scaled_layers[i].calculate_gradients(split_results[i].clone()));
        }
        
        let mut q_gradients: DMatrix<f64> = self.prev_k.clone();
        let mut k_gradients: DMatrix<f64> = self.prev_k.clone();
        let mut v_gradients: DMatrix<f64> = self.prev_k.clone();

        for i in 0..self.number_of_heads {
            q_gradients += self.q_linear[i].calculate_gradients(concat_gradients[i].0.clone());
            k_gradients += self.k_linear[i].calculate_gradients(concat_gradients[i].1.clone());
            v_gradients += self.v_linear[i].calculate_gradients(concat_gradients[i].2.clone());
        }

        (q_gradients, k_gradients, v_gradients)
       }
    
    pub fn adjust_parameters(&mut self, learning_rate: f64) {
        for i in 0..self.number_of_heads {
            self.q_linear[i].adjust_parameters(learning_rate);
            self.k_linear[i].adjust_parameters(learning_rate);
            self.v_linear[i].adjust_parameters(learning_rate);
        }

        self.final_linear.adjust_parameters(learning_rate);
    }
}


pub struct Decoder {
    pub norm_1: LayerNorm,
    pub norm_2: LayerNorm,
    pub mha: MultiHeadAttention,
    pub ffn: FFN
}

impl Decoder {
    pub fn new(d_model: usize, number_of_heads: usize, ffn_inner_size: usize) -> Decoder {
        Decoder {
            norm_1: LayerNorm::new(1.0),
            norm_2: LayerNorm::new(1.0),
            mha: MultiHeadAttention::new(d_model, number_of_heads),
            ffn: FFN::new(d_model, ffn_inner_size)
        }
    }

    pub fn calculate(&mut self, input: DMatrix<f64>, mask: DMatrix<f64>) -> DMatrix<f64> {
        let normal_matrix_1: DMatrix<f64> = self.norm_1.calculate(input.clone());
        let mha_result: DMatrix<f64> = input + self.mha.calculate(normal_matrix_1.clone(), normal_matrix_1.clone(), normal_matrix_1.clone(), mask);

        let normal_matrix_2: DMatrix<f64> = self.norm_2.calculate(mha_result);
        let ffn_result: DMatrix<f64> = self.ffn.calculate(normal_matrix_2);

        ffn_result
    }

    pub fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>) -> DMatrix<f64> {
        let ffn_output_gradient: DMatrix<f64> = self.ffn.calculate_gradients(previous_gradient.clone());
        let normal_2_gradients: DMatrix<f64> = self.norm_2.calculate_gradients(ffn_output_gradient);
        
        let mha_gradients: (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) = self.mha.calculate_gradients(normal_2_gradients);
        let normal_1_gradients: DMatrix<f64> = self.norm_1.calculate_gradients(mha_gradients.0 + mha_gradients.1 + mha_gradients.2);
        
        normal_1_gradients + previous_gradient
    }

    pub fn adjust_parameters(&mut self, learning_rate: f64) {
        self.norm_1.adjust_parameters(learning_rate);
        self.norm_2.adjust_parameters(learning_rate);
        self.ffn.adjust_parameters(learning_rate);
        self.mha.adjust_parameters(learning_rate);
    }
}