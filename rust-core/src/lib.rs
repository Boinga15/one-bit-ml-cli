use pyo3::prelude::*;

#[pyfunction]
fn square_odd_reduce_even(number: i32) -> PyResult<i32> {
    if number % 2 == 0 {
        return Ok(0);
    }

    Ok(number * number)
}

#[pymodule]
fn onebitml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(square_odd_reduce_even, m)?)?;
    Ok(())
}