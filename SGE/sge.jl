# include the SPICE module
include("SPICE/src/SPICE.jl")
using SPICE

# ---- add the State-change matrix
v = [1 0; -1 0; 0 1; 0 -1]

# ---- add the Initial state for the species
x0 = [0,0]

# ---- Bounds on the initial (log) search space for parameters
bounds = [[1e-6,1e-6,1e-6],
		[10.0,10.0,10.0]]


# ---- We specify seperable mass-action propensity functions (acting on a path p, containing as attributes the species x, parameters θ, propensity vector a, time t)
function F(p)
    H(p)
    p.a .*= p.θ
end

function H(p)
    p.a[1] = 1 - p.x[1]
    p.a[2] = p.x[1] 
    p.a[3] = p.x[1]
    p.a[4] = p.x[2]
end

# ---- Specify the model to SPICE, passing information that the observables in the data are called X1 and X2.
model = Model(x0, F, H, v, bounds, obsname=[:X1,:X2])

# ---- We can tweak the algorithm parameters below
# e.g., ssa = :Tau (or :Direct)
cem = CEM(ssa=:Tau, nElite = 10, nRepeat=1, nSamples=1000, maxIter=250, mSamples=20000, shoot=false, splitting=false, sampling=:log, tauOpt=TauOpt(ϵ=0.1))

# ---- Initialise the system, and point it to the folder containing the data
system = System(model, "./sim0", routine=cem)

# ---- run 1 run of the algorithm
estimate(system, 1, "sge")
