# Dense coupled solve for the condensed solver: a ComplexF32 LU of the ComplexF64 system with
# iterative refinement to the Float64 backward error.

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
end

function RefinedDenseLU(matrix::Matrix{ComplexF64})
    all(isfinite, matrix) || throw(ArgumentError("dense coupled matrix has non-finite entries"))
    refined = RefinedDenseLU(matrix, opnorm(matrix, Inf), nothing, nothing, 0, nothing, NaN)
    if maximum(abs, matrix; init=0.0) > floatmax(Float32)
        _dense_fall_back!(refined, "an entry is outside the Float32 range")
    else
        candidate = lu!(ComplexF32.(matrix); check=false)
        if issuccess(candidate) && all(isfinite, candidate.factors)
            refined.factor = candidate
        else
            _dense_fall_back!(refined, "the Float32 factorization is singular or not finite")
        end
    end
    return refined
end

function _dense_fall_back!(factorization::RefinedDenseLU, reason::AbstractString)
    factorization.fallback_reason = "Float32 LU with refinement fell back to a Float64 LU: " * reason
    @warn factorization.fallback_reason
    factorization.factor = nothing
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
    mul!(residual, factorization.matrix, solution, -one(ComplexF64), one(ComplexF64))
    return residual
end

function Base.:\(factorization::RefinedDenseLU, rhs::AbstractVecOrMat)
    all(isfinite, rhs) || throw(ArgumentError("dense coupled right-hand side has non-finite entries"))
    target = ComplexF64.(rhs)
    residual = similar(target)
    if isnothing(factorization.fallback)
        solution = ComplexF64.(factorization.factor \ ComplexF32.(target))
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
            solution .+= ComplexF64.(factorization.factor \ ComplexF32.(residual))
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
