"""
    BeatEngineMumps

Optional sequential MUMPS (`MUMPS_seq_jll`) Schur-complement backend for the condensed coupled
solver's FEM interior.

Nothing here is a hard dependency. Only the macOS Metal environment (`julia_metal`) lists
`MUMPS_seq_jll` and `OpenBLAS32_jll`, so CPU, CUDA and ROCm installs never download their
artifacts; in those environments `mumps_library()` reports that the package is absent. The JLL is
required lazily by package id from the active environment the first time a MUMPS condensation is
requested; if it is absent or fails to load, if the loaded
library is not the MUMPS version whose C struct this file mirrors, or if the load-time
self-test disagrees with a dense reference, `mumps_library()` reports the reason and the caller
factors with UMFPACK instead.

Binding: a direct `ccall` to `zmumps_c` with `ZMumpsStruc`, a field-for-field mirror of
`ZMUMPS_STRUC_C` from MUMPS 5.9.1's `zmumps_c.h` (32-bit `MUMPS_INT`, 64-bit `MUMPS_INT8`).
Julia lays out an immutable-field `mutable struct` with C alignment; the size and a spread of
field offsets are asserted against the values a C compiler reports for that header, and the
version string (which sits past every array in the struct) is read back after `JOB=-1`.

BLAS: `MUMPS_seq_jll` links `libblastrampoline` and calls the LP64 (32-bit integer) BLAS
interface, which Julia's default configuration does not populate (ILP64 OpenBLAS only). The
loader forwards `OpenBLAS32_jll` into the LP64 slots without clearing Julia's ILP64 backend, and
sets that library's thread pool directly before each MUMPS call, so `BLAB_MUMPS_THREADS` does
not change the thread count of Julia's own dense LU. The sequential MUMPS build has no OpenMP;
its parallelism is the BLAS inside the frontal factorization.
"""
module BeatEngineMumps

using LinearAlgebra, SparseArrays
import Base.Libc.Libdl

export mumps_library, MumpsSchurSolver, mumps_analyse!, mumps_factorize!, mumps_reduce,
    mumps_expand, mumps_interior_solve, mumps_release!, mumps_threads

const MUMPS_SEQ_PKGID = Base.PkgId(Base.UUID("d7ed1dd3-d0ae-5e8e-bfb4-87a502085b8d"), "MUMPS_seq_jll")
const OPENBLAS32_PKGID = Base.PkgId(Base.UUID("656ef2d0-ae68-5445-9ca0-591084a874a2"), "OpenBLAS32_jll")
const MUMPS_LAYOUT_VERSION = "5.9.1"
const USE_COMM_WORLD = Int32(-987654)

"""Mirror of `ZMUMPS_STRUC_C` (MUMPS 5.9.1, `MUMPS_INTSIZE32`)."""
mutable struct ZMumpsStruc
    sym::Int32
    par::Int32
    job::Int32
    comm_fortran::Int32
    icntl::NTuple{60,Int32}
    keep::NTuple{500,Int32}
    cntl::NTuple{15,Float64}
    dkeep::NTuple{230,Float64}
    keep8::NTuple{150,Int64}
    n::Int32
    nblk::Int32
    nz_alloc::Int32
    nz::Int32
    nnz::Int64
    irn::Ptr{Int32}
    jcn::Ptr{Int32}
    a::Ptr{ComplexF64}
    nz_loc::Int32
    nnz_loc::Int64
    irn_loc::Ptr{Int32}
    jcn_loc::Ptr{Int32}
    a_loc::Ptr{ComplexF64}
    nelt::Int32
    eltptr::Ptr{Int32}
    eltvar::Ptr{Int32}
    a_elt::Ptr{ComplexF64}
    blkptr::Ptr{Int32}
    blkvar::Ptr{Int32}
    perm_in::Ptr{Int32}
    sym_perm::Ptr{Int32}
    uns_perm::Ptr{Int32}
    colsca::Ptr{Float64}
    rowsca::Ptr{Float64}
    colsca_from_mumps::Int32
    rowsca_from_mumps::Int32
    colsca_loc::Ptr{Float64}
    rowsca_loc::Ptr{Float64}
    rowind::Ptr{Int32}
    colind::Ptr{Int32}
    pivots::Ptr{ComplexF64}
    rhs::Ptr{ComplexF64}
    redrhs::Ptr{ComplexF64}
    rhs_sparse::Ptr{ComplexF64}
    sol_loc::Ptr{ComplexF64}
    rhs_loc::Ptr{ComplexF64}
    rhsintr::Ptr{ComplexF64}
    irhs_sparse::Ptr{Int32}
    irhs_ptr::Ptr{Int32}
    isol_loc::Ptr{Int32}
    irhs_loc::Ptr{Int32}
    glob2loc_rhs::Ptr{Int32}
    glob2loc_sol::Ptr{Int32}
    nrhs::Int32
    lrhs::Int32
    lredrhs::Int32
    nz_rhs::Int32
    lsol_loc::Int32
    nloc_rhs::Int32
    lrhs_loc::Int32
    nsol_loc::Int32
    schur_mloc::Int32
    schur_nloc::Int32
    schur_lld::Int32
    mblock::Int32
    nblock::Int32
    nprow::Int32
    npcol::Int32
    ld_rhsintr::Int32
    info::NTuple{80,Int32}
    infog::NTuple{80,Int32}
    rinfo::NTuple{40,Float64}
    rinfog::NTuple{40,Float64}
    pivnul_list::Ptr{Int32}
    mapping::Ptr{Int32}
    singular_values::Ptr{Float64}
    nb_singular_values::Int32
    size_schur::Int32
    listvar_schur::Ptr{Int32}
    schur::Ptr{ComplexF64}
    wk_user::Ptr{ComplexF64}
    version_number::NTuple{32,UInt8}
    ooc_tmpdir::NTuple{1024,UInt8}
    ooc_prefix::NTuple{256,UInt8}
    write_problem::NTuple{1024,UInt8}
    lwk_user::Int32
    save_dir::NTuple{1024,UInt8}
    save_prefix::NTuple{256,UInt8}
    metis_options::NTuple{40,Int32}
    instance_number::Int32
    function ZMumpsStruc()
        s = new()
        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), pointer_from_objref(s), 0, sizeof(ZMumpsStruc))
        return s
    end
end

# `offsetof`/`sizeof` as reported by a C compiler for MUMPS 5.9.1's zmumps_c.h on 64-bit
# macOS/Linux. A mismatch means this mirror is wrong for the platform, which would be silent
# memory corruption; refuse the library instead.
const C_LAYOUT = (
    sizeof=10920,
    icntl=16, keep=256, cntl=2256, dkeep=2376, keep8=4216, n=5416, nnz=5432, irn=5440, a=5456,
    colsca_from_mumps=5592, rowind=5616, rhs=5640, nrhs=5736, ld_rhsintr=5796, info=5800,
    infog=6120, rinfog=6760, nb_singular_values=7104, size_schur=7108, listvar_schur=7112,
    schur=7120, wk_user=7128, version_number=7136, lwk_user=9472, metis_options=10756,
    instance_number=10916,
)

function layout_mismatches()
    mismatches = String[]
    sizeof(ZMumpsStruc) == C_LAYOUT.sizeof ||
        push!(mismatches, "sizeof $(sizeof(ZMumpsStruc)) != $(C_LAYOUT.sizeof)")
    for name in keys(C_LAYOUT)
        name == :sizeof && continue
        offset = fieldoffset(ZMumpsStruc, Base.fieldindex(ZMumpsStruc, name))
        offset == getfield(C_LAYOUT, name) ||
            push!(mismatches, "$name at $offset != $(getfield(C_LAYOUT, name))")
    end
    return mismatches
end

_field_pointer(s::ZMumpsStruc, name::Symbol, ::Type{T}) where {T} =
    Ptr{T}(pointer_from_objref(s) + fieldoffset(ZMumpsStruc, Base.fieldindex(ZMumpsStruc, name)))

function set_icntl!(s::ZMumpsStruc, index::Integer, value::Integer)
    GC.@preserve s unsafe_store!(_field_pointer(s, :icntl, Int32), Int32(value), index)
    return s
end
icntl(s::ZMumpsStruc, index::Integer) = s.icntl[index]
infog(s::ZMumpsStruc, index::Integer) = s.infog[index]
version_string(s::ZMumpsStruc) = String(UInt8[c for c in s.version_number if c != 0x00]) |> strip

struct MumpsLibrary
    available::Bool
    reason::String
    zmumps_c::Ptr{Cvoid}
    set_blas_threads::Ptr{Cvoid}
    version::String
end

const LIBRARY = Ref{Union{Nothing,MumpsLibrary}}(nothing)
const LIBRARY_LOCK = ReentrantLock()
#: Test hook: pretend the library is missing, to exercise the UMFPACK fallback.
const FORCE_UNAVAILABLE = Ref(false)
#: Live solvers, released at process exit if their owner did not release them.
const LIVE_SOLVERS = WeakKeyDict{Any,Nothing}()
const ATEXIT_REGISTERED = Ref(false)

"""`BLAB_MUMPS_THREADS`, default 4: the LP64 OpenBLAS pool size used inside MUMPS calls."""
function mumps_threads()
    raw = strip(get(ENV, "BLAB_MUMPS_THREADS", "4"))
    value = tryparse(Int, raw)
    (isnothing(value) || value < 1) &&
        error("Unsupported BLAB_MUMPS_THREADS value: $raw. Expected a positive integer.")
    return value
end

function _unavailable(reason)
    return MumpsLibrary(false, reason, C_NULL, C_NULL, "")
end

"""
    mumps_library() -> MumpsLibrary

Load (once per process) and self-test the MUMPS binding. Never throws: failure is returned as
`available=false` with a reason. Call it from a single task before any concurrent host BLAS work
starts on first use, because forwarding the LP64 backend edits libblastrampoline's tables.
"""
function mumps_library()
    FORCE_UNAVAILABLE[] && return _unavailable("disabled by test hook")
    cached = LIBRARY[]
    isnothing(cached) || return cached
    lock(LIBRARY_LOCK) do
        isnothing(LIBRARY[]) || return LIBRARY[]
        LIBRARY[] = try
            _load_library()
        catch exception
            _unavailable(sprint(showerror, exception))
        end
        return LIBRARY[]
    end
end

function _load_library()
    mismatches = layout_mismatches()
    isempty(mismatches) || return _unavailable("ZMUMPS_STRUC_C mirror mismatch: " * join(mismatches, "; "))
    for pkgid in (MUMPS_SEQ_PKGID, OPENBLAS32_PKGID)
        isnothing(Base.locate_package(pkgid)) &&
            return _unavailable("$(pkgid.name) is not in this Julia environment (only the Metal environment ships MUMPS)")
    end
    mumps_module = Base.require(MUMPS_SEQ_PKGID)
    openblas_module = Base.require(OPENBLAS32_PKGID)
    Base.invokelatest(getproperty, mumps_module, :is_available) |> Base.invokelatest ||
        return _unavailable("MUMPS_seq_jll has no artifact for this platform")
    Base.invokelatest(getproperty, openblas_module, :is_available) |> Base.invokelatest ||
        return _unavailable("OpenBLAS32_jll has no artifact for this platform")
    mumps_path = Base.invokelatest(getproperty, mumps_module, :libzmumps_path)
    openblas_path = Base.invokelatest(getproperty, openblas_module, :libopenblas_path)

    config = BLAS.get_config()
    if !any(library -> library.interface == :lp64, config.loaded_libs)
        BLAS.lbt_forward(openblas_path; clear=false)
        any(library -> library.interface == :lp64, BLAS.get_config().loaded_libs) ||
            return _unavailable("could not forward an LP64 BLAS for MUMPS")
    end
    openblas_handle = Libdl.dlopen(openblas_path)
    set_threads = Libdl.dlsym_e(openblas_handle, :openblas_set_num_threads)
    mumps_handle = Libdl.dlopen(mumps_path)
    zmumps_c = Libdl.dlsym(mumps_handle, :zmumps_c)

    library = MumpsLibrary(true, "", zmumps_c, set_threads, "")
    version, error_message = _self_test(library)
    isempty(error_message) || return _unavailable("self-test failed: " * error_message)
    if !ATEXIT_REGISTERED[]
        atexit(_release_live_solvers)
        ATEXIT_REGISTERED[] = true
    end
    return MumpsLibrary(true, "", zmumps_c, set_threads, version)
end

function _release_live_solvers()
    for solver in collect(keys(LIVE_SOLVERS))
        try
            mumps_release!(solver)
        catch
        end
    end
    return nothing
end

function _call!(library::MumpsLibrary, s::ZMumpsStruc, job::Integer)
    s.job = Int32(job)
    GC.@preserve s ccall(library.zmumps_c, Cvoid, (Ptr{ZMumpsStruc},), pointer_from_objref(s))
    return s
end

"""
    MumpsSchurSolver

One MUMPS SYM=2 (complex-symmetric LDLᵀ) instance with a centralized Schur complement on a fixed
variable list. Holds the analysed pattern so a frequency sweep refactors numerically only, and
the lower-triangle extraction map from the caller's full CSC matrix.
"""
mutable struct MumpsSchurSolver
    library::MumpsLibrary
    struc::ZMumpsStruc
    initialized::Bool
    n::Int
    colptr::Vector{Int}
    rowval::Vector{Int}
    lower_positions::Vector{Int}
    mirror_positions::Vector{Int}
    irn::Vector{Int32}
    jcn::Vector{Int32}
    values::Vector{ComplexF64}
    schur_variables::Vector{Int32}
    schur_buffer::Vector{ComplexF64}
    threads::Int
    analysed::Bool
    factored::Bool
    analysis_count::Int
    factorization_count::Int
end

function MumpsSchurSolver(library::MumpsLibrary; threads::Int=mumps_threads())
    library.available || error("MUMPS is unavailable: $(library.reason)")
    s = ZMumpsStruc()
    s.sym = 2
    s.par = 1
    s.comm_fortran = USE_COMM_WORLD
    _call!(library, s, -1)
    infog(s, 1) < 0 && error("MUMPS initialization failed: INFOG(1)=$(infog(s, 1)) INFOG(2)=$(infog(s, 2))")
    solver = MumpsSchurSolver(library, s, true, 0, Int[], Int[], Int[], Int[], Int32[], Int32[],
        ComplexF64[], Int32[], ComplexF64[], threads, false, false, 0, 0)
    _quiet!(s)
    LIVE_SOLVERS[solver] = nothing
    # A solver the owner forgot still returns its factors to the allocator eventually.
    finalizer(mumps_release!, solver)
    return solver
end

function _quiet!(s::ZMumpsStruc)
    set_icntl!(s, 1, -1)   # error messages
    set_icntl!(s, 2, -1)   # diagnostics
    set_icntl!(s, 3, -1)   # global information
    set_icntl!(s, 4, 0)    # print level
    set_icntl!(s, 5, 0)    # assembled input
    set_icntl!(s, 6, 0)    # no value-based column permutation: the analysis stays pattern-only
    set_icntl!(s, 7, 7)    # automatic ordering (METIS on these matrices)
    set_icntl!(s, 8, 0)    # no scaling
    set_icntl!(s, 12, 1)   # plain symmetric ordering, no value-based 2x2 compression
    set_icntl!(s, 18, 0)   # centralized matrix
    set_icntl!(s, 19, 3)   # centralized Schur complement, by columns
    set_icntl!(s, 20, 0)   # dense right-hand sides
    set_icntl!(s, 21, 0)   # centralized solution
    # Test (BLAB_TEST_MUMPS_ICNTL="7=5,35=2", BLAB_TEST_MUMPS_CNTL="7=1e-12"): overrides, applied
    # before the analysis.
    for (name, setter) in (("BLAB_TEST_MUMPS_ICNTL", (i, v) -> set_icntl!(s, i, parse(Int, v))),
                           ("BLAB_TEST_MUMPS_CNTL", (i, v) -> _test_set_cntl!(s, i, parse(Float64, v))))
        for item in split(get(ENV, name, ""), ','; keepempty=false)
            index, value = split(strip(item), '=')
            setter(parse(Int, index), value)
        end
    end
    return s
end

_test_get_cntl(s::ZMumpsStruc, index::Integer) =
    GC.@preserve s unsafe_load(_field_pointer(s, :cntl, Float64), index)

_test_set_cntl!(s::ZMumpsStruc, index::Integer, value::Real) =
    GC.@preserve s unsafe_store!(_field_pointer(s, :cntl, Float64), Float64(value), index)

function _check(solver::MumpsSchurSolver, phase::AbstractString)
    status = infog(solver.struc, 1)
    status < 0 && error("MUMPS $phase failed: INFOG(1)=$status INFOG(2)=$(infog(solver.struc, 2))")
    return nothing
end

"""
`BLAB_MUMPS_SOLVE_THREADS`, default 1: the pool size for solve phases (`JOB=3`). Those are
triangular sweeps over a few right-hand sides; on SAWMOD 8 threads made them ~4x slower than 2.
"""
function mumps_solve_threads()
    raw = strip(get(ENV, "BLAB_MUMPS_SOLVE_THREADS", "1"))
    value = tryparse(Int, raw)
    (isnothing(value) || value < 1) &&
        error("Unsupported BLAB_MUMPS_SOLVE_THREADS value: $raw. Expected a positive integer.")
    return value
end

function _set_blas_threads(solver::MumpsSchurSolver, count::Int=solver.threads)
    solver.library.set_blas_threads == C_NULL && return nothing
    ccall(solver.library.set_blas_threads, Cvoid, (Cint,), Cint(count))
    return nothing
end

"""
    mumps_analyse!(solver, matrix, schur_variables) -> Bool

Analyse `matrix`'s pattern (lower triangle) with a Schur complement on `schur_variables`, unless
the pattern and variable list match what was already analysed. Returns whether the previous
analysis was reused.
"""
function mumps_analyse!(solver::MumpsSchurSolver, matrix::SparseMatrixCSC, schur_variables::AbstractVector{<:Integer})
    solver.initialized || error("MUMPS solver was released.")
    n = size(matrix, 1)
    size(matrix, 2) == n || error("MUMPS Schur condensation needs a square matrix.")
    variables = Int32.(schur_variables)
    if solver.analysed && solver.n == n && solver.colptr == matrix.colptr &&
       solver.rowval == matrix.rowval && solver.schur_variables == variables
        return true
    end
    rows = rowvals(matrix)
    lower = Int[]
    mirror = Int[]
    irn = Int32[]
    jcn = Int32[]
    sizehint!(lower, nnz(matrix) ÷ 2 + n)
    for column in 1:n, position in nzrange(matrix, column)
        row = rows[position]
        row >= column || continue
        push!(lower, position)
        push!(irn, Int32(row))
        push!(jcn, Int32(column))
        # The transpose entry; its absence means an asymmetric pattern.
        range = nzrange(matrix, row)
        index = searchsortedfirst(view(rows, range), column)
        index <= length(range) && rows[range[index]] == column ||
            error("MUMPS SYM=2 needs a structurally symmetric matrix.")
        push!(mirror, range[index])
    end
    solver.n = n
    solver.colptr = copy(matrix.colptr)
    solver.rowval = copy(matrix.rowval)
    solver.lower_positions = lower
    solver.mirror_positions = mirror
    solver.irn = irn
    solver.jcn = jcn
    solver.values = zeros(ComplexF64, length(lower))
    solver.schur_variables = variables
    solver.schur_buffer = zeros(ComplexF64, length(variables)^2)
    s = solver.struc
    s.n = Int32(n)
    s.nnz = Int64(length(lower))
    s.nz = Int32(0)
    s.size_schur = Int32(length(variables))
    _bind_arrays!(solver)
    _set_blas_threads(solver)
    GC.@preserve solver _call!(solver.library, s, 1)
    _check(solver, "analysis")
    solver.analysed = true
    solver.factored = false
    solver.analysis_count += 1
    return false
end

function _bind_arrays!(solver::MumpsSchurSolver)
    s = solver.struc
    s.irn = pointer(solver.irn)
    s.jcn = pointer(solver.jcn)
    s.a = pointer(solver.values)
    s.listvar_schur = pointer(solver.schur_variables)
    s.schur = pointer(solver.schur_buffer)
    s.schur_lld = Int32(length(solver.schur_variables))
    return s
end

"""
    mumps_factorize!(solver, matrix; symmetry_tolerance) -> Matrix{ComplexF64}

Numeric LDLᵀ of the analysed pattern with `matrix`'s values and return the full Schur complement
`S = A_ΓΓ - A_ΓI A_II⁻¹ A_IΓ` in `schur_variables` order. Refuses (throws) a matrix whose
transpose entries differ by more than `symmetry_tolerance` relative, since SYM=2 reads only the
lower triangle.
"""
# Test hook (BLAB_TEST_HOST_POOL): `(T, m, n) -> Matrix{T}` for the Schur copy, or nothing.
const _TEST_SCHUR_TAKE = Ref{Any}(nothing)

function mumps_factorize!(solver::MumpsSchurSolver, matrix::SparseMatrixCSC; symmetry_tolerance::Real)
    solver.analysed || error("MUMPS factorization needs an analysis first.")
    values = nonzeros(matrix)
    for (index, position) in enumerate(solver.lower_positions)
        value = values[position]
        mirror = values[solver.mirror_positions[index]]
        if value != mirror
            abs(value - mirror) <= symmetry_tolerance * max(abs(value), abs(mirror)) ||
                error("MUMPS SYM=2 needs a complex-symmetric matrix; entry ($(solver.irn[index]), $(solver.jcn[index])) differs from its transpose.")
        end
        solver.values[index] = ComplexF64(value)
    end
    test_t0 = time_ns()
    fill!(solver.schur_buffer, zero(ComplexF64))
    s = solver.struc
    _bind_arrays!(solver)
    _set_blas_threads(solver)
    attempts = 0
    test_t1 = time_ns()
    while true
        GC.@preserve solver _call!(solver.library, s, 2)
        status = infog(s, 1)
        # -8/-9: workspace estimate too small; relax it and retry a bounded number of times.
        if status in (-8, -9) && attempts < 4
            set_icntl!(s, 14, max(icntl(s, 14), 20) * 2)
            attempts += 1
            continue
        end
        # Test (BLAB_TEST_MUMPS_CNTL="1=0", no numerical pivoting): a failed factorization goes back to
        # MUMPS's default threshold for this and every later factorization of the solver.
        if status < 0 && _test_get_cntl(s, 1) == 0.0
            _test_set_cntl!(s, 1, 0.01)
            @warn "MUMPS factorization without pivoting failed (INFOG(1)=$status); retrying with CNTL(1)=0.01"
            continue
        end
        break
    end
    _check(solver, "factorization")
    # Test (BLAB_TEST_MUMPS_STATS=<file>): one line per factorization with MUMPS's own statistics.
    test_stats = get(ENV, "BLAB_TEST_MUMPS_STATS", "")
    if !isempty(test_stats)
        open(test_stats, "a") do io
            println(io, "n=$(solver.n) nz=$(length(solver.values)) schur=$(length(solver.schur_variables)) ",
                    "prep_s=$((test_t1 - test_t0) / 1e9) fac_s=$((time_ns() - test_t1) / 1e9) ",
                    "rinfog1=$(s.rinfog[1]) rinfog3=$(s.rinfog[3]) infog9=$(infog(s, 9)) infog12=$(infog(s, 12)) ",
                    "infog13=$(infog(s, 13)) infog29=$(infog(s, 29)) infog11=$(infog(s, 11)) threads=$(solver.threads)")
        end
    end
    solver.factored = true
    solver.factorization_count += 1
    m = length(solver.schur_variables)
    # Test (BLAB_TEST_HOST_POOL): the parent module's pool supplies the copy's storage.
    take = _TEST_SCHUR_TAKE[]
    schur = isnothing(take) ? copy(reshape(solver.schur_buffer, m, m)) :
            copyto!(take(eltype(solver.schur_buffer), m, m), reshape(solver.schur_buffer, m, m))
    # MUMPS 5.9 returns the full matrix for SYM=2 with ICNTL(19)=3; older releases returned one
    # triangle. Complete it from the lower triangle if the strict upper part came back empty.
    if m > 1 && all(iszero, (schur[i, j] for j in 2:m for i in 1:(j-1))) &&
       any(!iszero, (schur[i, j] for j in 1:(m-1) for i in (j+1):m))
        for j in 2:m, i in 1:(j-1)
            schur[i, j] = schur[j, i]
        end
    end
    return schur
end

function _solve_phase!(solver::MumpsSchurSolver, rhs::Matrix{ComplexF64}, reduced::Matrix{ComplexF64}, mode::Integer)
    solver.factored || error("MUMPS solve needs a factorization first.")
    size(rhs, 1) == solver.n || error("MUMPS right-hand side must have one row per matrix row.")
    size(reduced) == (length(solver.schur_variables), size(rhs, 2)) ||
        error("MUMPS reduced right-hand side has the wrong shape.")
    size(rhs, 2) == 0 && return rhs
    s = solver.struc
    _bind_arrays!(solver)
    s.nrhs = Int32(size(rhs, 2))
    s.lrhs = Int32(solver.n)
    s.lredrhs = Int32(length(solver.schur_variables))
    set_icntl!(s, 26, mode)
    _set_blas_threads(solver, mumps_solve_threads())
    GC.@preserve solver rhs reduced begin
        s.rhs = pointer(rhs)
        s.redrhs = pointer(reduced)
        _call!(solver.library, s, 3)
        s.rhs = C_NULL
        s.redrhs = C_NULL
    end
    set_icntl!(s, 26, 0)
    _check(solver, "solve (ICNTL(26)=$mode)")
    return rhs
end

"""
    mumps_reduce(solver, rhs) -> Matrix{ComplexF64}

Condensation phase (`ICNTL(26)=1`): `b_Γ - A_ΓI A_II⁻¹ b_I` for full-length right-hand-side
columns `rhs`, in `schur_variables` order.
"""
function mumps_reduce(solver::MumpsSchurSolver, rhs::AbstractMatrix)
    work = Matrix{ComplexF64}(rhs)
    reduced = zeros(ComplexF64, length(solver.schur_variables), size(work, 2))
    _solve_phase!(solver, work, reduced, 1)
    return reduced
end

"""
    mumps_expand(solver, schur_solution) -> Matrix{ComplexF64}

Expansion phase (`ICNTL(26)=2`) for the right-hand sides of the *immediately preceding*
`mumps_reduce`: full columns whose interior rows are `A_II⁻¹ (b_I - A_IΓ x_Γ)`.

MUMPS 5.9 keeps the reduction's forward solution internally and ignores the `RHS` contents
passed to the expansion (measured: expanding after reducing `b1` with `b2` in `RHS` returns
the `b1` solution), and one expansion consumes it (a second fails with `INFOG(1)=-35`). The
condensed solver therefore back-substitutes with `mumps_interior_solve` on an explicit
right-hand side; this is kept for the self-test and for callers that pair the two calls.
"""
function mumps_expand(solver::MumpsSchurSolver, schur_solution::AbstractMatrix)
    solution = zeros(ComplexF64, solver.n, size(schur_solution, 2))
    reduced = Matrix{ComplexF64}(schur_solution)
    _solve_phase!(solver, solution, reduced, 2)
    return solution
end

"""
    mumps_interior_solve(solver, rhs) -> Matrix{ComplexF64}

Internal-problem solve (`ICNTL(26)=0` with a Schur factorization): full-length columns whose
interior rows are `A_II⁻¹ b_I` and whose Schur rows are zero. One forward/backward pass, no
dependence on earlier calls. The Schur rows of `rhs` are zeroed before the call.
"""
function mumps_interior_solve(solver::MumpsSchurSolver, rhs::AbstractMatrix)
    work = Matrix{ComplexF64}(rhs)
    work[solver.schur_variables, :] .= zero(ComplexF64)
    _solve_phase!(solver, work, zeros(ComplexF64, length(solver.schur_variables), size(work, 2)), 0)
    return work
end

"""Free the MUMPS instance (`JOB=-2`). Idempotent."""
function mumps_release!(solver::MumpsSchurSolver)
    solver.initialized || return nothing
    solver.initialized = false
    solver.analysed = false
    solver.factored = false
    s = solver.struc
    s.rhs = C_NULL
    s.redrhs = C_NULL
    _call!(solver.library, s, -2)
    delete!(LIVE_SOLVERS, solver)
    return nothing
end

"""
Dense-reference self-test on a small complex-symmetric system with a Schur block, run once when
the library loads: checks the version string, the Schur complement, reduction, expansion, a
pattern-reusing refactorization, and release. Returns `(version, error_message)`.
"""
function _self_test(library::MumpsLibrary)
    n = 12
    retained = [3, 7, 12]
    interior = setdiff(1:n, retained)
    matrix = spzeros(ComplexF64, n, n)
    for i in 1:n
        matrix[i, i] = 4.0 + 0.5im * i
        i < n && (matrix[i, i+1] = matrix[i+1, i] = -1.0 + 0.1im)
        i + 4 <= n && (matrix[i, i+4] = matrix[i+4, i] = 0.3 - 0.2im)
    end
    solver = MumpsSchurSolver(library; threads=1)
    try
        version = version_string(solver.struc)
        version == MUMPS_LAYOUT_VERSION ||
            return version, "library reports MUMPS version '$version', binding mirrors $MUMPS_LAYOUT_VERSION"
        dense = Matrix(matrix)
        reference(A) = A[retained, retained] - A[retained, interior] * (A[interior, interior] \ A[interior, retained])
        mumps_analyse!(solver, matrix, retained)
        schur = mumps_factorize!(solver, matrix; symmetry_tolerance=0.0)
        norm(schur - reference(dense)) <= 1e-12 * norm(reference(dense)) ||
            return version, "Schur complement disagrees with the dense reference"
        rhs = ComplexF64[sin(i) + im * cos(2j) for i in 1:n, j in 1:2]
        reduced = mumps_reduce(solver, rhs)
        expected_reduced = rhs[retained, :] - dense[retained, interior] * (dense[interior, interior] \ rhs[interior, :])
        norm(reduced - expected_reduced) <= 1e-12 * norm(expected_reduced) ||
            return version, "reduced right-hand side disagrees with the dense reference"
        x_gamma = schur \ reduced
        expanded = mumps_expand(solver, x_gamma)
        full = dense \ rhs
        norm(expanded[interior, :] - full[interior, :]) <= 1e-12 * norm(full) ||
            return version, "expanded interior solution disagrees with the dense reference"
        interior_solution = mumps_interior_solve(solver, rhs)
        expected_interior = dense[interior, interior] \ rhs[interior, :]
        norm(interior_solution[interior, :] - expected_interior) <= 1e-12 * norm(expected_interior) &&
            iszero(interior_solution[retained, :]) ||
            return version, "internal-problem solve disagrees with the dense reference"
        shifted = matrix + 0.25 * sparse(I, n, n)
        reused = mumps_analyse!(solver, shifted, retained)
        reused || return version, "an identical pattern was re-analysed"
        schur2 = mumps_factorize!(solver, shifted; symmetry_tolerance=0.0)
        norm(schur2 - reference(Matrix(shifted))) <= 1e-12 * norm(schur2) ||
            return version, "refactorization with a reused analysis disagrees with the dense reference"
        return version, ""
    finally
        mumps_release!(solver)
    end
end

end # module
