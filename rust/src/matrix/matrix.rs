use std::fmt::Debug;

use std::ops::Add;

use num_traits::Num;

#[derive(Clone)]
#[derive(Debug)]
pub struct Matrix<T: Num + Clone + Debug> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T> // 1D vector used instead of Vec<Vec<f32>> to maximise space efficiency.
}

impl<T: Num + Clone + Debug> Matrix<T> {
    pub fn new(rows: usize, cols: usize, init_val: T) -> Matrix<T> {
        Matrix {
            rows: rows,
            cols: cols,
            data: vec![init_val; rows * cols]
        }
    }

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
}

impl <T: Add<Output = T> + Num + Clone + Debug> Add for Matrix<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
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