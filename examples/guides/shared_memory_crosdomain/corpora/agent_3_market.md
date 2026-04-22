# Market ticker modeler — Asset-Q value across ticks

Each snippet below is an independent observation record. Cite by the
bracketed ID when producing findings.

## [market_01]

Asset-Q's observed value is sampled at discrete, equally spaced ticks. The
ratio of the value at one tick to the value at the previous tick is
recorded; the logarithm of this ratio is termed the tick-return. Across a
long sample window, tick-returns have a mean close to a fixed constant and
a spread that is stable across the window. The mean may be positive,
negative, or near zero depending on the asset, but it is approximately
constant across sub-windows.

## [market_02]

When tick-returns are aggregated across n consecutive ticks, the total
log-change from the initial value has a center that grows steadily with n
and a spread that also grows with n. Individual tick-returns show no
detectable autocorrelation: the tick-return at tick n carries no
information about the tick-return at tick n+1. Conditional on the current
value, earlier values contribute no detectable predictive information
about future values.

## [market_03]

Because Asset-Q's value is reconstructed by multiplying successive ratios
rather than by adding differences, the value itself remains strictly
positive across all ticks regardless of how many negative tick-returns
occur. The distribution of log-values at a fixed future tick, given the
current value, is single-peaked and roughly symmetric, while the
distribution of raw values is asymmetric because the log transform is
nonlinear. Equal-sized positive and negative tick-returns do not produce
equal-sized changes in the raw value.

## [market_04]

The future distribution of Asset-Q's value depends only on the current
value, not on the sequence of earlier values by which it arrived. Observed
at progressively finer sub-tick sampling, the same qualitative statistical
structure appears at every scale: log-changes are centered on a steadily
shifting trend, their spread grows with the number of elapsed ticks, and
log-increments across non-overlapping windows are statistically
independent. This structure is preserved under rescaling of the tick axis
and the log-value axis together, by appropriately matched factors.
