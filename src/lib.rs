//! # kabsch_umeyama
//!
//! The Kabsch-Umeyama algorithm is a method for aligning and comparing the similarity between two sets of points.
//! It finds the optimal translation, rotation and scaling by minimizing the root-mean-square deviation (RMSD) of the point pairs.
use nalgebra::{
    allocator::Allocator, Const, DMatrix, DVector, DefaultAllocator, Dim, DimDiff, DimMin, DimSub,
    SMatrix, U1,
};
use nalgebra_lapack::SVD;
use std::ops::{Deref, MulAssign};

pub type NestedArray<const R: usize, const C: usize> = [[f64; C]; R];
pub struct Array2<const R: usize, const C: usize>(NestedArray<R, C>);

impl<const R: usize, const C: usize> From<NestedArray<R, C>> for Array2<R, C> {
    fn from(nested_array: NestedArray<R, C>) -> Self {
        Self(nested_array)
    }
}

impl<const R: usize, const C: usize> From<&[f64]> for Array2<R, C> {
    fn from(slice: &[f64]) -> Self {
        if slice.len() != R * C {
            panic!("The lengths do not match!")
        }
        let mut nested_array = [[0.; C]; R];
        nested_array
            .as_flattened_mut()
            .into_iter()
            .zip(slice)
            .for_each(|(a, v)| *a = *v);
        Self(nested_array)
    }
}

impl<const R: usize, const C: usize, const RC: usize> From<&[f64; RC]> for Array2<R, C> {
    fn from(array: &[f64; RC]) -> Self {
        if RC != R * C {
            panic!("The lengths do not match!")
        }
        Self::from(array.as_slice())
    }
}

impl<const R: usize, const C: usize> Into<SMatrix<f64, R, C>> for Array2<R, C> {
    fn into(self) -> SMatrix<f64, R, C> {
        SMatrix::<f64, R, C>::from_row_slice(self.0.as_flattened())
    }
}

impl<const R: usize, const C: usize> Deref for Array2<R, C> {
    type Target = NestedArray<R, C>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const R: usize, const C: usize> Array2<R, C> {
    /// New Array2 from a nested array
    pub fn new(nested_array: NestedArray<R, C>) -> Self {
        Self(nested_array)
    }

    /// Number of rows
    pub const fn nrows(&self) -> usize {
        R
    }

    /// Number of columns
    pub const fn ncols(&self) -> usize {
        C
    }
}

/// Estimate a similarity transformation between two matrices (2 Dimensions) with or without scaling.
/// The `None` values are returned only if the problem is not well-conditioned.
/// # Examples
/// ```
/// use kabsch_umeyama::{Array2, estimate};
///
/// // create an array src with 2 rows and 3 columns from a nested array
/// let src = Array2::from([[1., 2., 3.], [4., 5., 6.]]);
///
/// // create a dst array with 2 rows and 3 columns from a reference array
/// let dst = Array2::<2, 3>::from(&[1., 2., 3., 4., 5., 6.]);
///
/// // estimate the translation matrix
/// let t = estimate(src, dst, true);
/// assert!(t.is_some())
/// ```
pub fn estimate<const R: usize, const C: usize>(
    src: impl Into<SMatrix<f64, R, C>>,
    dst: impl Into<SMatrix<f64, R, C>>,
    estimate_scale: bool,
) -> Option<DMatrix<f64>>
where
    Const<C>: DimMin<Const<C>, Output = Const<C>> + DimSub<U1> + Dim,
    DefaultAllocator: Allocator<DimDiff<Const<C>, U1>> + Allocator<Const<C>>,
{
    let mut src = src.into();
    let mut dst = dst.into();
    let num = R as f64;
    let src_mean = src.row_mean();
    let dst_mean = dst.row_mean();
    src.row_iter_mut().for_each(|mut row| {
        row.iter_mut()
            .zip(src_mean.data.as_slice())
            .for_each(|(v, mean)| *v -= *mean);
    });
    dst.row_iter_mut().for_each(|mut row| {
        row.iter_mut()
            .zip(dst_mean.data.as_slice())
            .for_each(|(v, mean)| *v -= *mean)
    });
    let src_demean = src;
    let dst_demean = dst;

    let a = dst_demean.transpose() * src_demean / num;
    let mut d = DVector::<f64>::from_element(C, 1.);

    if a.determinant() < 0. {
        d[C - 1] = -1.;
    }
    let mut t = DMatrix::from_diagonal(&DVector::<f64>::from_element(C + 1, 1.));
    if let Some(svd) = SVD::new(a) {
        let s = svd.singular_values;
        let v = svd.vt;
        let u = svd.u;

        let rank = a.rank(1e-5f64);
        if rank == 0 {
            return None;
        }
        let m = if rank == C - 1 {
            if u.determinant() * v.determinant() > 0. {
                u * v
            } else {
                let cache = d[C - 1];
                d[C - 1] = -1.;
                let d_diag = DMatrix::from_diagonal(&d);
                let m = u * d_diag * v;
                d[C - 1] = cache;
                m
            }
        } else {
            let d_diag = DMatrix::from_diagonal(&d);
            u * d_diag * v
        };
        t.view_mut((0, 0), (C, C)).copy_from_slice(m.as_slice());

        let scale = if estimate_scale {
            1. / src_demean.row_variance().sum() * s.dot(&d)
        } else {
            1.
        };
        let mx = dst_mean - (t.view((0, 0), (C, C)) * src_mean.transpose()).transpose() * scale;
        t.view_mut((0, C), (C, 1)).copy_from_slice(mx.as_slice());
        t.view_mut((0, 0), (C, C)).mul_assign(scale);
        Some(t)
    } else {
        None
    }
}