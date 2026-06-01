# PyAMG vs AMJax: Feature Analysis

## JAX constraints 

**1. Control flow must be known at compile time:**
JAX analyzes the entire function before running it and builds a fixed execution plan. Loops and conditionals whose behavior depends on the data cannot be expressed freely: everything must be determined at trace time.

**2. Data is immutable:**
Any modification to an array produces a new array so that the original is left unchanged. For algorithms that update data sequentially, this generates a chain of copies that the compiler cannot optimize efficiently.

**3. No side effects inside compiled functions:**
A JIT-compiled function must be fully self-contained: no writes to external variables, no print statements, no calls to arbitrary Python functions. Only numerical operations on JAX arrays are allowed.

---

## Feature table

| Feature | Implemented | Implementable | Justification |
|---|---|---|---|
| **SMOOTHERS** | | | |
| Jacobi | ✓ | — | — |
| Gauss-Seidel | ✗ | No | **Problem:** The update of `x_i` depends on `x_j^{k+1}` for `j < i`, which creates an irreducible sequential dependency.<br>**JAX implementation, but:** Implementation via `lax.scan` is possible but would produce an O(n) XLA graph, not vectorizable on GPU and so as slow (maybe even slower) than on a CPU. |
| SOR | ✗ | No | **Problem:** Gauss-Seidel with `omega`, same sequential dependency as GS.<br>**JAX implementation, but:** Same as GS, `lax.scan` would produce an O(n) XLA graph not vectorizable on GPU. |
| Block Gauss-Seidel | ✗ | No | **Problem:** Same structure as GS, by blocks, there is a sequential inter-block dependency.<br>**JAX implementation, but:** Same as GS, `lax.scan` would produce an O(n) XLA graph not vectorizable on GPU. |
| Block Jacobi | ✗ | Yes | Variables are partitioned into independent blocks. Each block update depends only on its own residual, so all blocks can be solved simultaneously. The block inverses are precomputed once at setup as small dense matrices. |
| Additive Schwarz | ✗ | Yes | Local solves are all independent, so all subdomains can be solved simultaneously.<br>**JAX implementation, but:** Subdomains can have different sizes. Since JAX requires fixed array shapes, all subdomains must be padded to the size of the largest one, and a boolean mask used to ignore the padded entries during computation. |
| Multiplicative Schwarz | ✗ | No | **Problem:** Each local solve modifies `x`, which changes `r = b - Ax` for the next subdomain and results in a sequential intra-sweep dependency.<br>**JAX implementation, but:** `lax.scan` over subdomains would produce an O(n) XLA graph, not vectorizable on GPU. |
| Richardson | ✗ | Yes | Each iteration computes `r = b - Ax` and updates `x += omega * r`. This is just one matrix-vector product and vector additions: fully parallel on GPU. `omega = c / rho(A)` is fixed and precomputed at setup. |
| Chebyshev | ✗ | Yes | Horner's rule: `d` sequential matmuls with `d` static: unrolled at JAX tracing, each matmul parallel. Coefficients precomputed from Chebyshev roots over spectrum interval `[a, b]`. |
| Jacobi-NE | ✗ | Yes | `x += omega * A^H * D_{AA^H}^{-1} * (b - Ax)`: same structure as Jacobi, fully parallel. `D_{AA^H}` precomputed at setup. |
| Gauss-Seidel-NE | ✗ | No | **Problem:** GS applied to normal equations `A A^H y = b`: sequential row by row.<br>**JAX implementation, but:** Same as GS, `lax.scan` would produce an O(n) XLA graph, not vectorizable on GPU. |
| Gauss-Seidel-NR | ✗ | No | **Problem:** GS applied to `A^H A x = A^H b`, sequential column by column.<br>**JAX implementation, but:** Same as GS, `lax.scan` would produce an O(n) XLA graph, not vectorizable on GPU. |
| CF-Jacobi / FC-Jacobi | ✗ | Yes | Jacobi applied separately on C-points then F-points (or reverse). Within a group, independent: `jnp.where` on static C/F mask. Mask available from PyAMG hierarchy via `lvl.splitting`. |
| CG as smoother | ✗ | Yes | `jax.scipy.sparse.linalg.cg` exists: with fixed static `maxiter`, a `lax.fori_loop` of CG steps is jit-compilable. |
| GMRES as smoother | ✗ | Yes | `jax.scipy.sparse.linalg.gmres` exists: same reasoning as CG. |
| CGNE / CGNR as smoother | ✗ | Yes (effort) | Absent from `jax.scipy`: require custom JAX implementation. Algorithmically simple (CG on normal equations). |
| **COARSE SOLVERS** | | | |
| jacobi | ✓ | — | — |
| pinv | ✓ | — | — |
| lu | ✓ | — | — |
| qr | ✓ | — | — |
| Cholesky | ✗ | Yes | `jax.scipy.linalg.cho_factor` / `cho_solve` exist. Same pattern as LU already implemented. Requires SPD matrix. |
| pinv2 | ✗ | Yes | pinv2 is just pinv with a rcond parameter to control SVD truncation. `jnp.linalg.pinv` already supports rcond, so the current implementation covers both (as it already covers pinv). |
| splu (sparse LU) | ✗ | No | **Problem:** `splu` relies on SuperLU, an external C library. JAX has no native sparse LU factorization: XLA simply doesn't support this operation.<br>**JAX implementation, but:** Calling SuperLU via `jax.pure_callback` is technically possible, but breaks `vmap` and `grad`. |
| CG | ✗ | Yes | `jax.scipy.sparse.linalg.cg` with fixed `maxiter`: jit-compilable. |
| GMRES | ✗ | Yes | `jax.scipy.sparse.linalg.gmres` with fixed `max_iter`: jit-compatible. |
| BiCGSTAB | ✗ | Yes | `jax.scipy.sparse.linalg.bicgstab` with fixed `maxiter`: jit-compilable. |
| FGMRES | ✗ | Yes (very high effort) | **Problem:** Flexible GMRES requires a variable preconditioner at each iteration.<br>**JAX implementation possible, but:** `jax.scipy.sparse.linalg.gmres` doesn't support this and a custom implementation would be very costly. |
| **CYCLES** | | | |
| V-cycle | ✓ | — | — |
| W-cycle | ✓ | — | — |
| F-cycle | ✓ | — | — |
| AMLI | ✗ | Partial | The cycle itself is implementable but PyAMG requires `accel='fgmres'` for AMLI since the preconditioner is nonlinear, without FGMRES in JAX, the cycle cannot be used in practice. |
| **KRYLOV ACCELERATION (via `jax.scipy.sparse.linalg` solvers + `aspreconditioner()`)** | | | |
| CG | ✓ | — | `jax.scipy.sparse.linalg.cg(A, b, M=ml.aspreconditioner())` |
| GMRES | ✓ | — | `jax.scipy.sparse.linalg.gmres(A, b, M=ml.aspreconditioner())` |
| BiCGSTAB | ✓ | — | `jax.scipy.sparse.linalg.bicgstab(A, b, M=ml.aspreconditioner())` |
| FGMRES | ✗ | Yes (very high effort) | Absent from `jax.scipy`. **Problem:** Requires a variable preconditioner at each Krylov iteration.<br>**JAX implementation, but:** A custom implementation would be very costly. |
| **OTHER** | | | |
| Complex matrices | ✗ | Yes | JAX supports `complex64`/`complex128` natively. Required adjustments. |
| Residual tracking under jit | ✗ | No | **Problem:** `lax.while_loop` allows no Python side effects: `list.append` is impossible inside the XLA graph.<br>Possible in debug mode only via `jax.debug.callback`, but incompatible with `vmap` and `grad`. |
