# Kabsch-Umeyama &emsp; [![Build Status]][actions] [![Latest Version]][crates.io]

[Build Status]: https://img.shields.io/github/actions/workflow/status/dat58/kabsch_umeyama/rust.yml?branch=main
[actions]: https://github.com/dat58/kabsch_umeyama/actions?query=branch%3Amain
[Latest Version]: https://img.shields.io/crates/v/kabsch_umeyama.svg
[crates.io]: https://crates.io/crates/kabsch_umeyama

**The Kabsch-Umeyama algorithm is a method for aligning and comparing the similarity between two sets of points. It finds the optimal translation, rotation and scaling by minimizing the root-mean-square deviation (RMSD) of the point pairs.**

---

## Features
- Efficient and accurate implementation of the Kabsch-Umeyama algorithm.
- Calculates translation, and (optional) scaling matrices (RxC dimensions).
- Suitable for various applications involving point cloud alignment.

---

## Dependency
If you have Ubuntu, follow the command below:
```shell
sudo apt install gfortran cmake
```
If you encounter an error, please check the requirements at [nalgebra-lapack](https://docs.rs/nalgebra-lapack/latest/nalgebra_lapack/)

---

## Examples
```rust
use kabsch_umeyama::{Array2, estimate};

fn main() {
    // create an array src with 2 rows and 3 columns from a nested array
    let src = Array2::from([[1., 2., 3.], [4., 5., 6.]]);

    // create a dst array with 2 rows and 3 columns from a slice
    let dst = Array2::<2, 3>::from(&[2., 3., 4., 5., 6., 7.]);
    
    // estimate the translation matrix
    let t = estimate(src, dst, true);
    println!("The homogeneous similarity transformation matrix is: {}", t);
}
```

## References
- [Least-squares estimation of transformation parameters between two point patterns](https://web.stanford.edu/class/cs273/refs/umeyama.pdf)
- [A Purely Algebraic Justification of the Kabsch-Umeyama Algorithm](https://arxiv.org/pdf/1902.03138)