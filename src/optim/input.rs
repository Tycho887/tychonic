use std::ops::Deref;

/// Explicit ownership of immutable estimator inputs, without reference counting.
#[derive(Debug)]
pub enum InputSource<'a, T> {
    Owned(T),
    Borrowed(&'a T),
}

impl<'a, T> From<T> for InputSource<'a, T> {
    fn from(value: T) -> Self {
        Self::Owned(value)
    }
}
impl<'a, T> From<&'a T> for InputSource<'a, T> {
    fn from(value: &'a T) -> Self {
        Self::Borrowed(value)
    }
}
impl<T> Deref for InputSource<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            Self::Owned(value) => value,
            Self::Borrowed(value) => value,
        }
    }
}
