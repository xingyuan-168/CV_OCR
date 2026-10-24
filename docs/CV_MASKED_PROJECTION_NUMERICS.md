# Masked projection candidate bound

The spatial fast path applies to at most1024 visible pixels and1 or3 channels.
It changes candidate exclusion only. Final public scores still use the existing
integer accumulators, float conversion and compatibility clamp. Larger supports
retain the existing complete masked path. No candidate count or time limit is
used to stop a search.

Let t and i be the full visible vectors, with a separate mean removed from each
channel. Let E=||t||². For each channel, sort visible template values, pair the
lowest with the highest, then the next lowest with the next highest. A pixel
component is used at most once. A pair's unit vector has entries−1/sqrt(2) and
+1/sqrt(2) in that channel. These directions are orthonormal and perpendicular
to channel constants. Reordering directions by template energy preserves this.

For any prefix of directions, use unscaled integer differences dt and di:

- B=sum(dt²)=2||Pt||².
- A=sum(di²)=2||Pi||².
- C=sum(dt*di)=2<Pt,Pi>.

Orthogonal decomposition gives an unknown remaining source component. Maximizing
its contribution with Cauchy–Schwarz over its unknown norm yields, for A>0,

`rho <= sqrt((max(C,0)²/A + 2E - B)/(2E))`.

For A=0, use `sqrt((2E-B)/(2E))`. A zero prefix source projection can therefore
be rejected once its template directions account for enough energy. It need
not be assumed that the rest of the image or mask is constant.

For an effective positive threshold tau, define `D=B-(1-tau²)*2E`. A prefix
with D<=0 cannot reject a candidate. Otherwise reject only if A=0 or
`max(C,0)² < A*D`. Negative C uses zero as an upper bound, not its absolute value.
Strict inequality preserves boundary candidates. All directions are actual
visible pixels; ignored pixels do not enter this inequality.

## Numerical enclosure

There are at most1536 disjoint channel pairs. Every integer contribution has
magnitude at most65025, so A, B and C have magnitude below100 million and fit
signed32 without saturation or wrap. Scalar code promotes before multiplication.
No mean subtraction is needed for these moments, so low-variance inputs do not
cause cancellation in A or C.

Template energy uses the original per-channel integer moments. Its additional
absolute error allowance is16*DBL_EPSILON times the sum of raw squares plus a
per-channel unit. The rejection budget subtracts twice that allowance and
64*DBL_EPSILON*(B+E+1), covering its division/subtraction and budget arithmetic.
The bounded1024-pixel path keeps the integer products below2^53.

The scalar comparison raises its left side and lowers its right side with
16*DBL_EPSILON factors. The AVX2 comparison converts exact integer A/C tofloat.
Its budget is reduced by32*FLT_EPSILON before conversion. The combined relative
errors of integer conversion, squaring, budget conversion and multiplication
are smaller than this outward margin. D<=its absolute allowance is retained.
All operands here are finite and bounded; no subnormal-range positive budgets
arise from the byte-valued template energy at this support size.

The threshold for exclusion is `max(0, requested-1.5e-6)`. This covers the
existing1e-6 snap to1, float rounding, and the public double scorer's arithmetic.
For this path each raw channel moment is below2^26 and its integer products are
exact indouble. A division/subtraction pair contributes at most2^-27 absolute
error; three channels and two additions give at most7*2^-27 for a complete
moment. Any nonzero integer-sample centered energy is at least1/2. Propagating
these bounds through normalization and float rounding is below the additional
0.5e-6 allowance. The final requested threshold is unchanged.

Constant templates and empty supports keep their existing exact-score behavior;
they do not enter this projection path. Whole-frame per-channel constants can
be rejected only for the nonconstant template path and a positive threshold.
The implementation proves this condition by byte equality, not sampled values.

## Verification and limits

Independent exhaustive tests cover constant and near-constant signals, positive
affine matches, sparse masks, all-transparent templates, threshold-adjacent
floats, scalar/AVX2 dispatch, ignored-background changes, changing allocations,
boundaries and public result semantics. Numerical tests complement the enclosure
above; measured maximum FFT errors are not used as a safety argument.

Planar source pixels are cached only while the outer byte comparison establishes
identical frame content. Any content, mode or tile change invalidates them.
Their memory is included in workspace retained-byte accounting and discard.
This proof does not assert5ms performance; the production full-call benchmark
and every required architecture/case/round determine that separately.

## 2026-10-24 implementation update

The <=1024-visible-pixel masked path now uses orthogonal disjoint-pair projection
bounds and planar AVX2 loads. It avoids a full-frame FFT and rejects constant
source projections without enumerating thousands of zero-energy exact checks.
The existing integer final score and threshold remain authoritative. Larger
supports retain the complete FFT fallback. See CV_MASKED_PROJECTION_NUMERICS.md.

Broader regressions exposed a separate old ordinary two-template bottleneck:
serial full-frame three-channel transforms and two inverse/normalization passes.
Whole-frame parallelization and four strips were measured and rejected because
changing-frame P95 still exceeded5ms. The candidate uses eight overlapping tiles
(4 columns x2 rows), independent IPP plans, and at most7 helpers per leased
workspace. Every legal score position is written exactly once; final local peaks,
exact scores, IoU filtering and output ordering still execute sequentially.
Templates too large for bounded overlap, small images, grayscale, single-template
and actual transparent-mask calls retain their respective existing paths.

All helpers are joined before the DLL call returns, including exception unwinding.
No persistent DLL-owned threads, affinity changes, scheduling priority changes or
new runtime dependencies are introduced. Each workspace owns all tile scratch,
score matrices and cached frame transforms; these count toward the existing idle
budget and are released on exception/eviction. Byte changes invalidate the tile
cache; changing template extents invalidates its geometry. Template spectra still
retain only one geometry per template. Odd-size seam/right-bottom placement and
single/pair/odd-batch cache switches are checked against independent placements
and the OpenCV reference. The old64-frame threshold-adjacent changing regression
also compares complete result text with its reference in test builds.

Production timing additionally includes all7 old ordinary cases, separately from
the13 transparent cases, each with64warmups/1000samples/3rounds and nearest-rank
statistics. Packaging now requires both reports for each architecture. Internal
parallelism is included in the complete-call timing; concurrent-call pressure
latency is reported separately from the single-caller5ms requirement.
