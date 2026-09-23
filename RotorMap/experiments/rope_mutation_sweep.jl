# ==============================================================================
# rope_mutation_sweep.jl -- inner-product decay of the s5m1c5 rope encoding
# under a 1000-point error-rate sweep on a RANDOM fragment, with a saved plot.
#
# PROBLEM
#   rope_decay.jl measures the decay on one adversarial PERIODIC fragment over
#   a coarse 0.01-step grid.  This is the baseline companion: a RANDOM
#   genome-like fragment, 1000 mutated copies at error rates EVENLY SPACED on
#   [0.0, 0.5] (the sampler's exact mutation model), inner products of each
#   mutated encoding against the pristine one, and a saved scatter of inner
#   product (x) vs error rate (y).
#
# SETUP
#   * fragment: k = 20,000 random ACGT bases (seed 42, generate_reference);
#   * encoder: RopeEncoder(k = 20_000, s = 5, m = 1, c = 5) -- the production
#     s5m1c5 config; float64 CPU reference _ref_rope_frag_real
#     (encode/reference.jl -- same math the GPU kernel + index save mirror),
#     normalize = 0 (unit complex energy, the flow's mode); split [Re; Im]
#     vector of length 2 * 4^5 = 2,048;
#   * sweep: err_i = 0.5*(i-1)/999 for i = 1..1000; mutation i uses
#     mutate(frag, err_i; rng = Xoshiro(seed + i)) -- deterministic per point
#     (err = edit-OPERATION fraction: 1/3 substitutions, 1/3 insertions,
#     1/3 deletions, ceil(k*err) ops total, length kept exactly k);
#   * scores vs the pristine encoding e0: the real dot of the two unit-energy
#     split vectors == Re<z0, zmut> -- exactly what the fp8 GEMM + top-1
#     computes; plus the complex modulus |<z0, zmut>| (phase-rotation
#     tolerant under ins/del).  Both live in [-1, 1] / [0, 1].
#
# OUTPUT (experiments/out/, gitignored; copied back to the dev checkout)
#   rope_mutation_sweep.csv   err,ip_re,ip_abs -- one row per mutated copy
#   rope_mutation_sweep.svg   side-by-side two-panel scatter (real fidelity,
#                             complex modulus), error rate on the y axis,
#                             20-bin means, annotated with the Pearson r --
#                             pure-Base SVG writer (no plotting package)
#
# NOTES
#   * CPU-only: the float64 reference encoder avoids the GPU's fp16/fp8
#     quantization noise confounding the sweep; 1000 x 20k-base encodes are
#     seconds across threads (each point writes only its own slot, so the
#     threaded sweep stays deterministic).
#   * no Statistics dep in the project -- Pearson r is computed by hand.
#   * write_sweep_svg is shared: rope_double_mutation_sweep.jl includes this
#     file (its main() is PROGRAM_FILE-guarded) and overrides title/subtitle/
#     xmax/legend_text.
#
# Run:  julia --project=. RotorMap/experiments/rope_mutation_sweep.jl
# ==============================================================================

inc(p...) = include(joinpath(@__DIR__, "..", p...))

inc("common", "treearrays.jl") # TreeArrays module (dna.jl's mutate needs it)
inc("common", "dna.jl") # generate_reference, mutate (the sampler's exact model)
inc("encode", "reference.jl") # _ref_rope_frag_real (+ ropeencoder.jl) -- pure Base

using Random
using LinearAlgebra
using Printf

# ---- configuration -----------------------------------------------------------
const RMS_K = 20_000                    # fragment length (== the encoder's k)
const RMS_S, RMS_M, RMS_C = 5, 1, 5     # the s5m1c5 encoder config
const RMS_N = 1000                      # mutated copies (== sweep resolution)
const RMS_SEED = 42                     # fragment + per-mutation rng seeds
const RMS_OUT = joinpath(@__DIR__, "out")

"""Split [Re; Im] unit-energy ropes -> (Re<z0,z1>, |<z0,z1>|)."""
function rope_inner(e0::AbstractVector{<:Real}, e1::AbstractVector{<:Real})
    D = length(e0) >> 1
    ipre = dot(e0, e1) # == Re<z0,z1> for the kernel's split layout
    ipim = dot(view(e0, 1:D), view(e1, D+1:2D)) -
           dot(view(e0, D+1:2D), view(e1, 1:D))
    return ipre, abs(ComplexF64(ipre, ipim))
end

function main()
    mkpath(RMS_OUT)
    re = RopeEncoder(k = RMS_K, s = RMS_S, m = RMS_M, c = RMS_C)
    frag = generate_reference(RMS_K; seed = RMS_SEED)
    e0, n0 = _ref_rope_frag_real(frag, re; normalize = 0)
    errs = range(0.0, 0.5; length = RMS_N)
    ipre = Vector{Float64}(undef, RMS_N)
    ipabs = Vector{Float64}(undef, RMS_N)

    @info "sweep" k = RMS_K s = RMS_S m = RMS_M c = RMS_C n = RMS_N seed = RMS_SEED
    @info "pristine encoding" energy = dot(e0, e0) norm = n0[1]
    t = @elapsed Threads.@threads for i in 1:RMS_N
        mut = mutate(frag, errs[i]; rng = Xoshiro(RMS_SEED + i))
        em, _ = _ref_rope_frag_real(mut, re; normalize = 0)
        ipre[i], ipabs[i] = rope_inner(e0, em)
    end
    @info "sweep done" seconds = round(t; digits = 1)

    @printf("%5s  %12s  %12s\n", "err", "Re<IP>", "|IP|")
    for i in 1:50:RMS_N # every 50th point + the err = 0.5 endpoint
        @printf("%5.3f  %12.6f  %12.6f\n", errs[i], ipre[i], ipabs[i])
    end
    @printf("%5.3f  %12.6f  %12.6f\n", errs[end], ipre[end], ipabs[end])

    csv = joinpath(RMS_OUT, "rope_mutation_sweep.csv")
    open(csv, "w") do io
        println(io, "err,ip_re,ip_abs")
        for i in 1:RMS_N
            println(io, errs[i], ",", ipre[i], ",", ipabs[i])
        end
    end

    svg = joinpath(RMS_OUT, "rope_mutation_sweep.svg")
    write_sweep_svg(svg, collect(errs), ipre, ipabs)

    @info "outputs" csv svg
end

# ---- pure-Base SVG scatter (no plotting package in the project deps) ---------

_pearson(x::AbstractVector, y::AbstractVector) =
    (n = length(x); mx = sum(x) / n; my = sum(y) / n;
    sxy = sum((x[i] - mx) * (y[i] - my) for i in 1:n);
    sxx = sum((x[i] - mx)^2 for i in 1:n);
    syy = sum((y[i] - my)^2 for i in 1:n);
    sxy / sqrt(sxx * syy))

function _nice_step(span::Real, target::Int)
    raw = span / max(target, 1)
    mag = 10.0^floor(log10(raw))
    for f in (1.0, 2.0, 2.5, 5.0, 10.0)
        f * mag >= raw - 1e-12 && return f * mag
    end
    return 10.0 * mag
end

_tickvals(lo::Real, hi::Real, step::Real) = collect(range(
    ceil(lo / step - 1e-9) * step, floor(hi / step + 1e-9) * step; step = step))

_fmt_tick(t::Real) =
    abs(t - round(t)) < 1e-9 ? string(Int(round(t))) : string(round(t; digits = 2))

function _svg_panel(io, x0, y0, w, h, xd, yd, xdata, ydata, xt, yt;
                    color::String, dark::String, title::String,
                    xlab::String, ylab::String,
                    nbins::Int, legend::Bool, xlabels::Bool, ylabels::Bool,
                    legend_text::String = "mutated copy (N=1000)")
    sx = x -> x0 + (x - xd[1]) / (xd[2] - xd[1]) * w
    sy = y -> y0 + h - (y - yd[1]) / (yd[2] - yd[1]) * h

    for t in xt # grid
        @printf(io, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"#e4e4e4\" stroke-width=\"1\"/>\n",
                sx(t), y0, sx(t), y0 + h)
    end
    for t in yt
        @printf(io, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"#e4e4e4\" stroke-width=\"1\"/>\n",
                x0, sy(t), x0 + w, sy(t))
    end
    if xd[1] < 0 < xd[2] # dashed zero reference(s)
        @printf(io, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"#bbbbbb\" stroke-width=\"1\" stroke-dasharray=\"5,4\"/>\n",
                sx(0.0), y0, sx(0.0), y0 + h)
    end
    if yd[1] < 0 < yd[2]
        @printf(io, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"#bbbbbb\" stroke-width=\"1\" stroke-dasharray=\"5,4\"/>\n",
                x0, sy(0.0), x0 + w, sy(0.0))
    end
    @printf(io, "<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"none\" stroke=\"#999999\" stroke-width=\"1\"/>\n",
            x0, y0, w, h)
    if ylabels # y tick labels (left panel only -- shared error-rate axis)
        for t in yt
            @printf(io, "<text x=\"%.2f\" y=\"%.2f\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#444444\">%s</text>\n",
                    x0 - 7, sy(t) + 4, _fmt_tick(t))
        end
    end
    if xlabels
        for t in xt
            @printf(io, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"#999999\" stroke-width=\"1\"/>\n",
                    sx(t), y0 + h, sx(t), y0 + h + 5)
            @printf(io, "<text x=\"%.2f\" y=\"%.2f\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#444444\">%s</text>\n",
                    sx(t), y0 + h + 18, _fmt_tick(t))
        end
        @printf(io, "<text x=\"%.2f\" y=\"%.2f\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#555555\">%s</text>\n",
                x0 + w / 2, y0 + h + 36, xlab)
    end

    pts = Tuple{Float64,Float64}[] # bin means over the sweep variable (drawn under the points)
    bwidth = (yd[2] - yd[1]) / nbins
    for b in 1:nbins
        blo = yd[1] + (b - 1) * bwidth
        bhi = blo + bwidth
        s = 0.0
        c = 0
        for i in eachindex(ydata)
            if (blo <= ydata[i] < bhi) || (b == nbins && ydata[i] == bhi)
                s += xdata[i]
                c += 1
            end
        end
        c == 0 || push!(pts, (sx(s / c), sy(blo + bwidth / 2)))
    end
    if !isempty(pts)
        print(io, "<polyline fill=\"none\" stroke=\"", dark, "\" stroke-width=\"2\" points=\"")
        print(io, join([@sprintf("%.2f,%.2f", p[1], p[2]) for p in pts], " "))
        println(io, "\"/>")
        for p in pts
            @printf(io, "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"2.6\" fill=\"%s\"/>\n", p[1], p[2], dark)
        end
    end

    for i in eachindex(ydata) # the sweep points
        @printf(io, "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"1.7\" fill=\"%s\" fill-opacity=\"0.30\"/>\n",
                sx(xdata[i]), sy(ydata[i]), color)
    end

    @printf(io, "<text x=\"%.2f\" y=\"%.2f\" font-family=\"sans-serif\" font-size=\"13\" font-weight=\"bold\" fill=\"#333333\">%s</text>\n",
            x0, y0 - 9, title)
    if ylabels
        @printf(io, "<text x=\"%.2f\" y=\"%.2f\" transform=\"rotate(-90 %.2f %.2f)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#555555\">%s</text>\n",
                x0 - 52, y0 + h / 2, x0 - 52, y0 + h / 2, ylab)
    end
    if legend # bottom-right: empty for these data (high err => low IP)
        lx, ly = x0 + w - 198, y0 + h - 30
        @printf(io, "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"3\" fill=\"%s\" fill-opacity=\"0.55\"/>\n", lx, ly, color)
        @printf(io, "<text x=\"%.2f\" y=\"%.2f\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#444444\">%s</text>\n",
                lx + 9, ly + 4, legend_text)
        @printf(io, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"%s\" stroke-width=\"2\"/>\n",
                lx - 1, ly + 20, lx + 21, ly + 20, dark)
        @printf(io, "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"2.6\" fill=\"%s\"/>\n", lx + 10, ly + 20, dark)
        @printf(io, "<text x=\"%.2f\" y=\"%.2f\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#444444\">20-bin mean</text>\n",
                lx + 27, ly + 24)
    end
end

function write_sweep_svg(path, errs::Vector{Float64}, ipre::Vector{Float64},
                         ipabs::Vector{Float64}; nbins::Int = 20, xmax::Real = 1.0,
                         title::String = "RoPE-DNA mutation error rate vs inner product (s5m1c5)",
                         subtitle::String = "random k=20,000 fragment &#183; 1000 mutated copies, error rate evenly spaced on [0.0, 0.5] &#183; seed $(RMS_SEED)",
                         legend_text::String = "mutated copy (N=1000)")
    W, H = 960, 640
    ml, mr, mt, mb, gap = 86, 28, 74, 66, 48
    pw = (W - ml - mr - gap) / 2 # side-by-side panels sharing the y axis
    ph = H - mt - mb
    yd = (0.0, 0.5) # error rate on the y axis (axes swapped)
    yt = _tickvals(yd..., _nice_step(yd[2] - yd[1], 10))
    xd1 = (min(-0.05, minimum(ipre)), xmax) # real fidelity may dip below 0
    xd2 = (0.0, xmax) # |IP| of unit-energy ropes is bounded by 1
    xt1 = _tickvals(xd1..., _nice_step(xd1[2] - xd1[1], 5))
    xt2 = _tickvals(xd2..., _nice_step(xd2[2] - xd2[1], 5))
    r_re = _pearson(errs, ipre)
    r_abs = _pearson(errs, ipabs)

    io = IOBuffer()
    println(io, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$(W)\" height=\"$(H)\" viewBox=\"0 0 $(W) $(H)\">")
    println(io, "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>")
    println(io, "<text x=\"$(W / 2)\" y=\"28\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"17\" font-weight=\"bold\" fill=\"#222222\">$(title)</text>")
    println(io, "<text x=\"$(W / 2)\" y=\"50\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#555555\">$(subtitle) &#183; Pearson r: Re = $(round(r_re; digits = 3)), |&#183;| = $(round(r_abs; digits = 3))</text>")

    _svg_panel(io, ml, mt, pw, ph, xd1, yd, ipre, errs, xt1, yt;
               color = "#1f77b4", dark = "#123f66",
               title = "Re&#10216;z&#8320;, zmut&#10217; -- real fidelity (the GEMM/search statistic)",
               xlab = "real inner product",
               ylab = "mutation error rate (edit-op fraction: 1/3 sub, 1/3 ins, 1/3 del)",
               nbins = nbins, legend = true, xlabels = true, ylabels = true,
               legend_text = legend_text)
    _svg_panel(io, ml + pw + gap, mt, pw, ph, xd2, yd, ipabs, errs, xt2, yt;
               color = "#d62728", dark = "#7c1416",
               title = "|&#10216;z&#8320;, zmut&#10217;| -- complex modulus (phase-rotation tolerant)",
               xlab = "complex modulus", ylab = "mutation error rate",
               nbins = nbins, legend = false, xlabels = true, ylabels = false,
               legend_text = legend_text)

    println(io, "</svg>")
    write(path, take!(io))
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
