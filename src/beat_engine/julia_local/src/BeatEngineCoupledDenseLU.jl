# Dense coupled solve for the condensed solver: a ComplexF32 LU of the ComplexF64 system with
# iterative refinement to the Float64 backward error, blocked LU and triangular solves, reuse of the
# previous frequency's factor as a GMRES preconditioner, and the flux elimination's `B_q W` product
# kept implicit.

const DENSE_REFINEMENT_MAX_ITERATIONS = 10

"""
    RefinedDenseLU(matrix)

Single-precision LU of a double-precision dense system, solved by iterative refinement:
`x ← x + F32⁻¹ (b - A x)` with the residual in `ComplexF64`. Each step contracts the error by
about `κ(A) eps(Float32)`; the Multi_region_SAWMOD systems (κ ≈ 5e6) converge in two steps.
The stopping test is LAPACK `zcgesv`'s backward error, `‖r‖∞ ≤ ‖x‖∞ ‖A‖∞ eps(Float64) √n` per
column (`‖A‖∞` the operator norm, `opnorm`), which is what a `ComplexF64` LU attains, whatever `κ`.
Convergence is tested before the stall rule, so a solve already at that level is accepted.

The `ComplexF64` LU is used instead, with the reason in `fallback_reason`, when
- an entry is outside the `Float32` range (narrowing would overflow),
- the `Float32` factorization is singular or not finite, or
- a solve stalls (the backward-error ratio fails to halve), produces a non-finite residual, or has
  not converged after `DENSE_REFINEMENT_MAX_ITERATIONS` steps.
The fallback factor is built once and serves every later solve. A matrix that is singular in
`Float64` too throws `SingularException`; non-finite matrices or right-hand sides throw
`ArgumentError`.

`matrix` is kept, not copied: the caller hands over ownership and must not modify it while the
factorization is in use (the coupled builder allocates it per system and never writes it again).
`iterations` is the most refinement steps any solve needed; `backward_error` is the worst accepted
column's ratio to the Float64 attainable backward error in the last solve (≤ 1 unless it came
from the fallback, whose ratio is recorded as computed).
"""
mutable struct RefinedDenseLU
    matrix::Matrix{ComplexF64}
    matrix_norm::Float64
    factor::Union{Nothing,LinearAlgebra.LU{ComplexF32,Matrix{ComplexF32},Vector{LinearAlgebra.BlasInt}}}
    fallback::Union{Nothing,LinearAlgebra.LU{ComplexF64,Matrix{ComplexF64},Vector{LinearAlgebra.BlasInt}}}
    iterations::Int
    fallback_reason::Union{Nothing,String}
    backward_error::Float64
    stale::Bool   # `factor` belongs to an earlier frequency (see `_STALE_DENSE_FACTOR`)
    correction::Any   # `B_q W` blocks kept out of `matrix` (see `_dense_mul!`), or nothing
end

# The flux elimination's `B_q W` columns stay out of the Float64 dense matrix. `correction.parts` holds them per block as factors (Float64 `coupling` and `schur`,
# plus the Float32 `coupling32` the BEM produced); every Float64 product (refinement residuals, stale
# GMRES) applies them exactly, and only the Float32 LU input gets them explicitly, from one cgemm per
# block (~4x zgemm). The Float32 accumulation error (2.7e-4 when it was in the Float64 matrix) then
# only slows the refinement, which still has to reach the Float64 backward error.
function _dense_mul!(y, factorization::RefinedDenseLU, x, alpha, beta)
    mul!(y, factorization.matrix, x, alpha, beta)
    correction = factorization.correction
    isnothing(correction) && return y
    for part in correction.parts
        gathered = x[part.columns, :]
        projected = similar(gathered, size(part.schur, 1), size(gathered, 2))
        _threaded_mul!(projected, part.schur, gathered, one(eltype(y)), zero(eltype(y)))
        _threaded_mul!(view(y, correction.rows, :), part.coupling, projected, alpha, one(eltype(y)))
    end
    return y
end

# `mul!` with the rows split over tasks, one BLAS call each: Accelerate barely threads a gemm with
# 1-3 columns (B*t 5.4 -> 2.0 ms, W*g 2.9 -> 0.9 ms on SAWMOD).
function _threaded_mul!(y, A, x, alpha, beta)
    parts = collect(Iterators.partition(axes(A, 1), cld(size(A, 1), Threads.nthreads())))
    Threads.@threads for rows in parts
        mul!(view(y, rows, :), view(A, rows, :), x, alpha, beta)
    end
    return y
end

function _scatter_correction!(matrix, correction, precision)
    for part in correction.parts
        if precision === Float32
            schur32 = _pool_converted(ComplexF32, part.schur)
            product = mul!(_pool_take(ComplexF32, size(part.coupling32, 1), size(schur32, 2)), part.coupling32, schur32)
            _pool_give!(schur32)
        else
            product = part.coupling * part.schur
        end
        for (local_column, column) in enumerate(part.columns)
            @views matrix[correction.rows, column] .+= product[:, local_column]
        end
        precision === Float32 && _pool_give!(product)
    end
    return matrix
end

# The Float32 LU input: `matrix` narrowed, plus the correction from single-precision products.
function _narrowed_matrix(refined)
    matrix = refined.matrix
    narrowed = _pool_take(ComplexF32, size(matrix)...)
    Threads.@threads for column in axes(matrix, 2)
        @inbounds for row in axes(matrix, 1)
            narrowed[row, column] = ComplexF32(matrix[row, column])
        end
    end
    isnothing(refined.correction) || _scatter_correction!(narrowed, refined.correction, Float32)
    return narrowed
end

# Row sums of |matrix| and its largest |entry|, threaded by row blocks (as `_dense_abs_stats`).
function _abs_row_sums(matrix::Matrix{<:Complex})
    m, n = size(matrix)
    sums = zeros(Float64, m)
    blocks = collect(Iterators.partition(1:m, 256))
    maxima = zeros(Float64, length(blocks))
    Threads.@threads for b in eachindex(blocks)
        largest = 0.0
        @inbounds for j in 1:n, i in blocks[b]
            a = abs(matrix[i, j])
            sums[i] += a
            largest = max(largest, a)
        end
        maxima[b] = largest
    end
    return sums, maximum(maxima; init=0.0)
end

# `|B_q| (|W| 1)` summed over the correction's blocks: per row of `correction.rows`, an upper bound on
# the row sums of |B_q W|, so the matrix norm without the product is bounded from above.
function _correction_row_bound(correction)
    total = nothing
    for part in correction.parts
        weights = vec(sum(abs, part.schur; dims=2))
        coupling = part.coupling
        bound = zeros(Float64, size(coupling, 1))
        blocks = collect(Iterators.partition(axes(coupling, 1), 256))
        Threads.@threads for b in eachindex(blocks)
            @inbounds for j in axes(coupling, 2), i in blocks[b]
                bound[i] += abs(coupling[i, j]) * weights[j]
            end
        end
        total = isnothing(total) ? bound : total .+ bound
    end
    return total
end

function _bounded_norm(matrix, correction)
    sums, max_entry = _abs_row_sums(matrix)
    bound = _correction_row_bound(correction)
    view(sums, correction.rows) .+= bound
    norm_bound = maximum(sums; init=0.0)
    return norm_bound, max(max_entry, maximum(bound; init=0.0))
end

# Adds the correction to `matrix` in Float64 (before a Float64 fallback or a dump).
function _materialize_correction!(refined)
    isnothing(refined.correction) && return refined
    _scatter_correction!(refined.matrix, refined.correction, Float64)
    refined.correction = nothing
    return refined
end

# The last fresh ComplexF32 factor of the request, reused as a GMRES preconditioner at later
# frequencies while the previous stale solve needed at most `DENSE_STALE_REUSE_MAX_ITERATIONS`
# iterations. A stale solve that has not reached the Float64 backward error after
# `DENSE_STALE_GMRES_MAX_ITERATIONS` iterations factors afresh and ends reuse for the rest of the
# request: a sweep ascends and the iterations grow with frequency, so on SAWMOD reuse covers
# 23 Hz - 0.9 kHz and saves the Float32 LU there. The driver resets it per request.
const DENSE_STALE_GMRES_MAX_ITERATIONS = 15
const DENSE_STALE_REUSE_MAX_ITERATIONS = 8

mutable struct StaleDenseFactor
    factor::Union{Nothing,LinearAlgebra.LU{ComplexF32,Matrix{ComplexF32},Vector{LinearAlgebra.BlasInt}}}
    last_iterations::Int
    disabled::Bool
end
const _STALE_DENSE_FACTOR = StaleDenseFactor(nothing, 0, false)

"""
    reset_condensed_request_state!()

Forget the reused dense factor and empty the host array pool. Called by the driver at the start and
end of every request, so neither carries over between requests.
"""
function reset_condensed_request_state!()
    _STALE_DENSE_FACTOR.factor = nothing
    _STALE_DENSE_FACTOR.last_iterations = 0
    _STALE_DENSE_FACTOR.disabled = false
    _pool_clear!()
    return nothing
end

# `opnorm(matrix, Inf)` and `maximum(abs, matrix)` in one threaded pass. `opnorm` walks the column-major matrix row by row (~95 ms at n = 3116); here each
# task owns a block of rows and sweeps it column by column, so every row sum adds the same terms
# in the same order (j = 1..n) as `opnorm` and both values are bit-identical.
function _dense_abs_stats(matrix::Matrix{<:Complex})
    m, n = size(matrix)
    block = 256
    blocks = cld(m, block)
    norms = zeros(Float64, blocks)
    maxima = zeros(Float64, blocks)
    Threads.@threads for b in 1:blocks
        rows = ((b - 1) * block + 1):min(b * block, m)
        sums = zeros(Float64, length(rows))
        largest = 0.0
        @inbounds for j in 1:n
            for (local_row, i) in enumerate(rows)
                a = abs(matrix[i, j])
                sums[local_row] += a
                largest = max(largest, a)
            end
        end
        norms[b] = isempty(sums) ? 0.0 : maximum(sums)
        maxima[b] = largest
    end
    return maximum(norms; init=0.0), maximum(maxima; init=0.0)
end

function _swap_rows!(A, ipiv, k, ke, columns)
    Threads.@threads for j in columns
        @inbounds for i in k:ke
            r = ipiv[i]
            r == i && continue
            A[i, j], A[r, j] = A[r, j], A[i, j]
        end
    end
end

# Right-looking blocked LU with partial pivoting whose trailing updates are single large `gemm!`
# calls. Accelerate's cgetrf reaches ~0.3-0.4 TFLOP/s at n = 3116, its cgemm ~2; this runs at ~0.6
# on an M1 Pro. Pivots can differ from getrf's on near ties, so the factor is not bit-identical;
# the refinement still has to reach the Float64 backward error.
const DENSE_LU_BLOCK = 512

function _blocked_lu!(A::Matrix{T}, nb::Int=DENSE_LU_BLOCK) where {T}
    n = LinearAlgebra.checksquare(A)
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, n)
    info = 0
    for k in 1:nb:n
        ke = min(k + nb - 1, n)
        _, panel_pivots, panel_info = LAPACK.getrf!(view(A, k:n, k:ke); check=false)
        info == 0 && panel_info > 0 && (info = k - 1 + panel_info)
        ipiv[k:ke] .= panel_pivots .+ (k - 1)
        _swap_rows!(A, ipiv, k, ke, 1:(k-1))
        ke < n || continue
        _swap_rows!(A, ipiv, k, ke, (ke+1):n)
        BLAS.trsm!('L', 'L', 'N', 'U', one(T), view(A, k:ke, k:ke), view(A, k:ke, (ke+1):n))
        BLAS.gemm!('N', 'N', -one(T), view(A, (ke+1):n, k:ke), view(A, k:ke, (ke+1):n), one(T),
                   view(A, (ke+1):n, (ke+1):n))
    end
    return LinearAlgebra.LU(A, ipiv, LinearAlgebra.BlasInt(info))
end

# The ComplexF32 LU solve as row swaps plus blocked triangular solves whose off-diagonal updates are
# `gemm!` calls. Accelerate's getrs barely threads: 15 ms for 3 right-hand sides at n = 3116, this
# 4 ms. Rounding differs from getrs (Float32 level); the refinement still has to reach the Float64
# backward error.
const DENSE_TRS_BLOCK = 128

function _lu_solve(factor::LinearAlgebra.LU{T}, rhs::AbstractVecOrMat; nb::Int=DENSE_TRS_BLOCK) where {T}
    B = Matrix{T}(reshape(rhs, size(rhs, 1), :))
    A = factor.factors
    n = size(A, 1)
    @inbounds for i in 1:n
        r = factor.ipiv[i]
        r == i && continue
        for c in axes(B, 2)
            B[i, c], B[r, c] = B[r, c], B[i, c]
        end
    end
    for k in 1:nb:n
        ke = min(k + nb - 1, n)
        BLAS.trsm!('L', 'L', 'N', 'U', one(T), view(A, k:ke, k:ke), view(B, k:ke, :))
        ke < n && BLAS.gemm!('N', 'N', -one(T), view(A, (ke+1):n, k:ke), view(B, k:ke, :), one(T),
                             view(B, (ke+1):n, :))
    end
    for ke in n:-nb:1
        k = max(ke - nb + 1, 1)
        BLAS.trsm!('L', 'U', 'N', 'N', one(T), view(A, k:ke, k:ke), view(B, k:ke, :))
        k > 1 && BLAS.gemm!('N', 'N', -one(T), view(A, 1:(k-1), k:ke), view(B, k:ke, :), one(T),
                            view(B, 1:(k-1), :))
    end
    return rhs isa AbstractVector ? vec(B) : B
end

function RefinedDenseLU(matrix::Matrix{ComplexF64}; correction=nothing)
    all(isfinite, matrix) || throw(ArgumentError("dense coupled matrix has non-finite entries"))
    refined = RefinedDenseLU(matrix, 0.0, nothing, nothing, 0, nothing, NaN, false, correction)
    stale = _STALE_DENSE_FACTOR
    reuse_stale = !stale.disabled && !isnothing(stale.factor) && size(stale.factor.factors) == size(matrix) &&
                  stale.last_iterations <= DENSE_STALE_REUSE_MAX_ITERATIONS
    if !isnothing(correction) && reuse_stale
        # No Float32 product for a stale factor: the norm is bounded from above instead, which only
        # makes the convergence test stricter by the bound's overshoot.
        matrix_norm, max_entry = _bounded_norm(matrix, correction)
        refined.matrix_norm = matrix_norm
        if max_entry <= floatmax(Float32)
            refined.factor = stale.factor
            refined.stale = true
            return refined
        end
    end
    # With a correction the norm comes from the Float32 LU input (the full matrix to ~1e-7).
    narrowed = isnothing(correction) ? nothing : _narrowed_matrix(refined)
    matrix_norm, max_entry = _dense_abs_stats(something(narrowed, matrix))
    refined.matrix_norm = Float64(matrix_norm)
    if max_entry > floatmax(Float32)
        _dense_fall_back!(refined, "an entry is outside the Float32 range")
    elseif reuse_stale
        refined.factor = stale.factor
        refined.stale = true
    else
        _fresh_factor!(refined, narrowed)
    end
    return refined
end

function _fresh_factor!(refined::RefinedDenseLU, narrowed=nothing)
    candidate = _blocked_lu!(something(narrowed, _narrowed_matrix(refined)))
    if issuccess(candidate) && all(isfinite, candidate.factors)
        refined.factor = candidate
        refined.stale = false
        previous = _STALE_DENSE_FACTOR.factor
        # The replaced stale factor is referenced by no system any more.
        isnothing(previous) || previous === candidate || _pool_give!(previous.factors)
        _STALE_DENSE_FACTOR.factor = candidate
        _STALE_DENSE_FACTOR.last_iterations = 0
    else
        _dense_fall_back!(refined, "the Float32 factorization is singular or not finite")
    end
    return refined
end

# Right-preconditioned GMRES (modified Gram-Schmidt, no restart) with the stale ComplexF32 factor, all right-hand sides in lockstep so the matvec and the
# preconditioner solve are batched. Stops when every column meets `RefinedDenseLU`'s Float64
# backward-error test on the true residual. Returns (solution, iterations, ratio) or nothing.
function _stale_gmres(factorization::RefinedDenseLU, target::Matrix{ComplexF64}, cap::Int)
    M = factorization.factor
    n, k = size(target)
    threshold = factorization.matrix_norm * eps(Float64) * sqrt(n)
    x0 = ComplexF64.(_lu_solve(M, ComplexF32.(target)))
    residual = copy(target)
    _dense_mul!(residual, factorization, x0, -one(ComplexF64), one(ComplexF64))
    ratio = _dense_backward_error_ratio(factorization, residual, x0)
    ratio <= 1 && return (x0, 0, ratio)
    beta = [norm(view(residual, :, c)) for c in 1:k]
    V = zeros(ComplexF64, n, cap + 1, k)
    Z = zeros(ComplexF64, n, cap, k)
    H = zeros(ComplexF64, cap + 1, cap, k)
    for c in 1:k
        beta[c] > 0 && (V[:, 1, c] .= view(residual, :, c) ./ beta[c])
    end
    basis = Matrix{ComplexF64}(undef, n, k)
    w = similar(target)
    trial = copy(x0)
    for j in 1:cap
        for c in 1:k
            basis[:, c] .= view(V, :, j, c)
        end
        preconditioned = ComplexF64.(_lu_solve(M, ComplexF32.(basis)))
        _dense_mul!(w, factorization, preconditioned, one(ComplexF64), zero(ComplexF64))
        estimate_ok = true
        for c in 1:k
            Z[:, j, c] .= view(preconditioned, :, c)
            beta[c] > 0 || continue
            wc = view(w, :, c)
            for i in 1:j
                h = dot(view(V, :, i, c), wc)
                H[i, j, c] = h
                wc .-= h .* view(V, :, i, c)
            end
            h = norm(wc)
            H[j+1, j, c] = h
            h > 0 && (V[:, j+1, c] .= wc ./ h)
            rhs = zeros(ComplexF64, j + 1)
            rhs[1] = beta[c]
            Hj = H[1:(j+1), 1:j, c]
            y = Hj \ rhs
            trial[:, c] .= view(x0, :, c) .+ view(Z, :, 1:j, c) * y
            norm(rhs - Hj * y) <= threshold * norm(view(trial, :, c), Inf) || (estimate_ok = false)
        end
        estimate_ok || continue
        copyto!(residual, target)
        _dense_mul!(residual, factorization, trial, -one(ComplexF64), one(ComplexF64))
        ratio = _dense_backward_error_ratio(factorization, residual, trial)
        isfinite(ratio) || return nothing
        ratio <= 1 && return (trial, j, ratio)
    end
    return nothing
end

function _dense_fall_back!(factorization::RefinedDenseLU, reason::AbstractString)
    factorization.fallback_reason = "Float32 LU with refinement fell back to a Float64 LU: " * reason
    @warn factorization.fallback_reason
    factorization.factor = nothing
    _materialize_correction!(factorization)
    factorization.fallback = lu(factorization.matrix)
    return factorization
end

Base.size(factorization::RefinedDenseLU, dims...) = size(factorization.matrix, dims...)

# Worst column's backward error, in units of the double-precision attainable one.
function _dense_backward_error_ratio(factorization::RefinedDenseLU, residual, solution)
    threshold = factorization.matrix_norm * eps(Float64) * sqrt(size(factorization.matrix, 1))
    return maximum(
        column -> norm(view(residual, :, column), Inf) /
                  max(norm(view(solution, :, column), Inf) * threshold, floatmin(Float64)),
        axes(solution, 2);
        init=0.0,
    )
end

function _dense_residual!(residual, factorization::RefinedDenseLU, target, solution)
    copyto!(residual, target)
    _dense_mul!(residual, factorization, solution, -one(ComplexF64), one(ComplexF64))
    return residual
end

function Base.:\(factorization::RefinedDenseLU, rhs::AbstractVecOrMat)
    all(isfinite, rhs) || throw(ArgumentError("dense coupled right-hand side has non-finite entries"))
    target = ComplexF64.(rhs)
    residual = similar(target)
    if isnothing(factorization.fallback) && factorization.stale
        result = _stale_gmres(factorization, Matrix(reshape(target, size(target, 1), :)), DENSE_STALE_GMRES_MAX_ITERATIONS)
        if !isnothing(result)
            solution, iterations, ratio = result
            _STALE_DENSE_FACTOR.last_iterations = iterations
            factorization.iterations = max(factorization.iterations, iterations)
            factorization.backward_error = ratio
            return rhs isa AbstractVector ? vec(solution) : solution
        end
        _STALE_DENSE_FACTOR.disabled = true
        _fresh_factor!(factorization)
    end
    if isnothing(factorization.fallback)
        solution = ComplexF64.(_lu_solve(factorization.factor, ComplexF32.(target)))
        previous = Inf
        reason = nothing
        for iteration in 0:DENSE_REFINEMENT_MAX_ITERATIONS
            ratio = _dense_backward_error_ratio(factorization, _dense_residual!(residual, factorization, target, solution), solution)
            if !isfinite(ratio)
                reason = "non-finite residual after $iteration refinement steps"
                break
            end
            if ratio <= 1
                factorization.backward_error = ratio
                return solution
            end
            iteration == DENSE_REFINEMENT_MAX_ITERATIONS && break
            if ratio > previous / 2
                reason = "refinement stalled at $(ratio)x the Float64 backward error after $iteration steps"
                break
            end
            previous = ratio
            solution .+= ComplexF64.(_lu_solve(factorization.factor, ComplexF32.(residual)))
            factorization.iterations = max(factorization.iterations, iteration + 1)
        end
        isnothing(reason) &&
            (reason = "refinement did not reach the Float64 backward error in $(DENSE_REFINEMENT_MAX_ITERATIONS) steps")
        _dense_fall_back!(factorization, reason)
    end
    solution = factorization.fallback \ target
    factorization.backward_error =
        _dense_backward_error_ratio(factorization, _dense_residual!(residual, factorization, target, solution), solution)
    return solution
end

"""
    dense_solver_diagnostics(system) -> Dict{String,Any}

The dense coupled factorization that ran: `dense_solver` (`lu_float32`, `lu_float64`,
`lu_float32_refined`, or `lu_float64_fallback` after a refinement fallback),
`dense_refinement_iterations`, `dense_refinement_fallback_reason` and `dense_refinement_backward_error`
(the last solve's worst column, in units of the Float64 attainable backward error).
"""
function dense_solver_diagnostics(system)
    factorization = hasproperty(system, :factorization) ? system.factorization : nothing
    if factorization isa RefinedDenseLU
        return Dict{String,Any}(
            "dense_solver" => isnothing(factorization.fallback) ? "lu_float32_refined" : "lu_float64_fallback",
            "dense_refinement_iterations" => factorization.iterations,
            "dense_refinement_fallback_reason" => factorization.fallback_reason,
            "dense_refinement_backward_error" => isnan(factorization.backward_error) ? nothing : factorization.backward_error,
        )
    end
    kind = factorization isa LinearAlgebra.LU ? "lu_" * lowercase(string(real(eltype(factorization)))) : nothing
    return Dict{String,Any}(
        "dense_solver" => kind,
        "dense_refinement_iterations" => 0,
        "dense_refinement_fallback_reason" => nothing,
        "dense_refinement_backward_error" => nothing,
    )
end
