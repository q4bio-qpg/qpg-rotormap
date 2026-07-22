using Pkg
# Pkg.add("Revise")
using Revise

Pkg.activate(joinpath(@__DIR__,".."))
using RotorMap 

@info "Loading app"

main()

function read_args()    
    include(joinpath(@__DIR__,"args.jl"))    
    new_args = invokelatest(args)
    @show new_args
    try 
        main(args=String.(split(new_args)), server=true)
    catch e 
        showerror(stdout, e, stacktrace(catch_backtrace()))
        println()
    end
end

# using Base.Threads
# @spawn while true 
#     sleep(1)
#     revise()
#     # @show Revise.revision_queue
# end

entr(read_args, [joinpath(@__DIR__,"args.jl")], postpone=true)


# julia --project=. -e 'using Pkg; Pkg.instantiate()'

# julia --project=. bin/run.jl -h