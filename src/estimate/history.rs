//! Filter history whose current record always exists.
use numeris::estimate::EkfStep;

/// Ordered records, with the current epoch stored independently of the archive.
pub struct FilterHistory<const N: usize> {
    completed: Vec<EkfStep<f64, N>>,
    current: EkfStep<f64, N>,
}

impl<const N: usize> FilterHistory<N> {
    pub(super) fn new(current: EkfStep<f64, N>) -> Self {
        Self {
            completed: Vec::new(),
            current,
        }
    }
    pub fn len(&self) -> usize {
        self.completed.len() + 1
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn get(&self, index: usize) -> Option<&EkfStep<f64, N>> {
        if index == self.completed.len() {
            Some(&self.current)
        } else {
            self.completed.get(index)
        }
    }
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &EkfStep<f64, N>> {
        self.completed.iter().chain(std::iter::once(&self.current))
    }
    /// Explicitly copy fixed-size records for Numeris's contiguous-slice API.
    pub fn to_records(&self) -> Vec<EkfStep<f64, N>> {
        self.iter().map(copy_record).collect()
    }
    pub(super) fn current_mut(&mut self) -> &mut EkfStep<f64, N> {
        &mut self.current
    }
    pub(super) fn push(&mut self, record: EkfStep<f64, N>) {
        self.completed
            .push(std::mem::replace(&mut self.current, record));
    }
    pub(super) fn checkpoint(&self) -> (usize, EkfStep<f64, N>) {
        (self.completed.len(), copy_record(&self.current))
    }
    pub(super) fn restore(&mut self, saved: (usize, EkfStep<f64, N>)) {
        self.completed.truncate(saved.0);
        self.current = saved.1;
    }
}

fn copy_record<const N: usize>(r: &EkfStep<f64, N>) -> EkfStep<f64, N> {
    EkfStep {
        x_predicted: r.x_predicted,
        p_predicted: r.p_predicted,
        x_updated: r.x_updated,
        p_updated: r.p_updated,
        f_jacobian: r.f_jacobian,
    }
}
