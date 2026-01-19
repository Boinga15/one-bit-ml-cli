use std::fmt::Debug;

use std::ops::{Add, Sub, Mul, Div};
use num_traits::{Num, Pow, Float};

pub trait Numeric: Num + Clone + Debug + Default + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Pow<Self, Output = Self> + Float {}

impl<T> Numeric for T
where 
    T: Num + Clone + Debug + Default + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Pow<Self, Output = Self> + Float,
{

}

#[derive(Clone)]
pub struct Matrix<T: Numeric> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T> // 1D vector used instead of Vec<Vec<f32>> to maximise space efficiency.
}


impl<T: Numeric> Matrix<T> {
    // Generation Functions
    pub fn new(rows: usize, cols: usize, init_val: T) -> Matrix<T> {
        Matrix {
            rows: rows,
            cols: cols,
            data: vec![init_val; rows * cols]
        }
    }


    // Helper Functions
    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[col + row * self.cols].clone()
    }

    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.data[col + row * self.cols] = value;
    }

    pub fn display(&self) {
        println!("Rows: {}, Cols: {}", self.rows, self.cols);
        println!("Data: {:?}", self.data);
    }


    // Raising to the power of either a single number of a matrix of the same size.
    pub fn pow_unit(&mut self, pow: T) -> Matrix<T> {
        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone().pow(pow.clone()));
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }

    pub fn pow_matrix(&mut self, other: Matrix<T>) -> Matrix<T> {
        if self.rows != other.rows || self.cols != other.cols {
            println!("ERROR ENCOUNTERED - Raising matrix to another matrix.");
            println!("Dimension mismatch: ({}, {})^({}, {})", self.rows, self.cols, other.rows, other.cols);
            return Matrix {
                rows: self.rows,
                cols: self.cols,
                data: self.data.clone()
            }; // Fallback option.
        }        

        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone().pow(other.data[i].clone()));
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }


    // Matrix Manipulation
    pub fn transpose(&self) -> Matrix<T> {
        let mut data: Vec<T> = vec![];

        for row in 0..self.rows {
            for col in 0..self.cols {
                data.push(self.get(col, row));
            }
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }


    // Element-wise Functions
    pub fn element_mult(&self, other: Matrix<T>) -> Matrix<T> {
        if self.rows != other.rows || self.cols != other.cols {
            println!("ERROR ENCOUNTERED - Multiplying two matrices element-wise.");
            println!("Dimension mismatch: ({}, {}) + ({}, {})", self.rows, self.cols, other.rows, other.cols);
            return Matrix {
                rows: self.rows,
                cols: self.cols,
                data: vec![]
            }; // Fallback option.
        }

        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone() * other.data[i].clone());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }

    // Standard Trigonometric Functions
    pub fn sin(&self) -> Matrix<T> {
        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone().sin());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }

    pub fn cos(&self) -> Matrix<T> {
        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone().cos());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }

    pub fn tan(&self) -> Matrix<T> {
        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone().tan());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }


    // Hyperbolic Functions
    pub fn sinh(&self) -> Matrix<T> {
        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone().sinh());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }

    pub fn cosh(&self) -> Matrix<T> {
        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone().cosh());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }

    pub fn tanh(&self) -> Matrix<T> {
        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone().tanh());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }


    // Normalization Functions
    pub fn softmax(&self) -> Matrix<T> {
        let mut total: T = T::default();

        for element in self.data.iter() {
            total = total + element.clone().exp();
        }

        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone().exp() / total);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }

    pub fn row_softmax(&self) -> Matrix<T> {
        let mut data: Vec<T> = vec![];

        for row in 0..self.rows {
            let mut total: T = T::default();

            for col in 0..self.cols {
                total = total + self.get(row, col).exp();
            }

            for col in 0..self.cols {
                data.push(self.get(row, col).clone().exp() / total);
            }
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }

    pub fn col_softmax(&self) -> Matrix<T> {
        let mut data: Vec<T> = vec![];

        for col in 0..self.cols {
            let mut total: T = T::default();

            for row in 0..self.rows {
                total = total + self.get(row, col).exp();
            }

            for row in 0..self.rows {
                data.push(self.get(row, col).clone().exp() / total);
            }
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }
}


// Add Functions
impl <T: Numeric> Add for Matrix<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            println!("ERROR ENCOUNTERED - Adding two matrices.");
            println!("Dimension mismatch: ({}, {}) + ({}, {})", self.rows, self.cols, other.rows, other.cols);
            return self; // Fallback option.
        }

        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone() + other.data[i].clone());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }
}

impl <T: Numeric> Add<T> for Matrix<T> {
    type Output = Self;

    fn add(self, other: T) -> Self {
        let data: Vec<T> = self.data.iter().map(|x| x.clone() + other.clone()).collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }
}


impl <T: Numeric> Sub for Matrix<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            println!("ERROR ENCOUNTERED - Subtracting two matrices.");
            println!("Dimension mismatch: ({}, {}) + ({}, {})", self.rows, self.cols, other.rows, other.cols);
            return self; // Fallback option.
        }

        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone() - other.data[i].clone());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }
}

impl <T: Numeric> Sub<T> for Matrix<T> {
    type Output = Self;

    fn sub(self, other: T) -> Self {
        let data: Vec<T> = self.data.iter().map(|x| x.clone() - other.clone()).collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }
}


impl <T: Numeric> Mul for Matrix<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.cols != other.rows {
            println!("ERROR ENCOUNTERED - Multiplying two matrices.");
            println!("Dimension mismatch: ({}, {}) * ({}, {})", self.rows, self.cols, other.rows, other.cols);
            return self; // Fallback option.
        }

        let mut data: Vec<T> = vec![];

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut total: T = T::default();

                for k in 0..self.cols {
                    total = total + self.get(i, k) * other.get(k, j);
                }

                data.push(total);
            }
        }

        Matrix {
            rows: self.rows,
            cols: other.cols,
            data: data
        }
    }
}

impl <T: Numeric> Mul<T> for Matrix<T> {
    type Output = Self;

    fn mul(self, other: T) -> Self {
        let data: Vec<T> = self.data.iter().map(|x| x.clone() * other.clone()).collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }
}


impl <T: Numeric> Div for Matrix<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            println!("ERROR ENCOUNTERED - Dividing two matrices.");
            println!("Dimension mismatch: ({}, {}) + ({}, {})", self.rows, self.cols, other.rows, other.cols);
            return self; // Fallback option.
        }

        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone() / other.data[i].clone());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }
}


impl <T: Numeric> Div<T> for Matrix<T> {
    type Output = Self;

    fn div(self, other: T) -> Self {
        let data: Vec<T> = self.data.iter().map(|x| x.clone() / other.clone()).collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }
}
