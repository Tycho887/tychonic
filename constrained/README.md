# tychonic-constrained

Fit a GCRF orbit and constant drag susceptibility to range-rate observations
with a fitted bias per pass and hard limits on antenna pointing. The Rust
ripopt solver accepts a prepared `ConstrainedRangeRateProblem` and returns
physical parameters, recovered biases and angular feasibility diagnostics.

See the [fitting guide](https://github.com/Tycho887/tychonic/blob/main/docs/constrained_range_rate.md)
for input units, closest-approach sampling and a usage example.
