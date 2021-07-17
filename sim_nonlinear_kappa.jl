using JLD2, FileIO, GraphIO, CSV, DataFrames
using Distributed
using Interpolations

_calc = false
_slurm = false

if _calc
    using ClusterManagers
	if length(ARGS) > 0
		N_tasks = parse(Int, ARGS[1])
	else
		N_tasks = 1
	end
    N_worker = N_tasks
	if _slurm
    	addprocs(SlurmManager(N_worker))
	else
		addprocs(N_worker)
	end
	println()
	println(nprocs(), " processes")
	println(length(workers()), " workers")
else
	using Plots
end

# here comes the broadcast
# https://docs.julialang.org/en/v1/stdlib/Distributed/index.html#Distributed.@everywhere
begin
	calc = $_calc # if false, only plotting
end

begin
	dir = @__DIR__
	include("$dir/src/system_structs.jl")
	include("$dir/src/network_dynamics.jl")
end

begin
		using DifferentialEquations
		using Distributions
		using LightGraphs
		using LinearAlgebra
		using Random
		using StatsBase
		using Statistics
		using Parameters
		using DSP
		using ToeplitzMatrices
		Random.seed!(42)
end




begin
	dir = @__DIR__
	N = 4 #Number of nodes
	batch_size = 1
	num_prod = 2 # producer nodes
	nom_cons = N - num_prod
	N_half = Int(N/2)
	num_days = 20
	l_day = 3600*24 # DemCurve.l_day
	l_hour = 3600 # DemCurve.l_hour
	l_minute = 60

end
begin
	current_filter = 1:Int(1.5N)
	voltage_filter = Int(1.5N)+1:Int(2.5N)
	energy_filter = Int(2.5N)+1:Int(3.5N)#3N+1:4N
end

begin
	graph = random_regular_graph(iseven(3N) ? N : (N-1), 3)

end
#_graph_lst = []
#for i in 1:1
#	push!(_graph_lst, random_regular_graph(iseven(3N) ? N : (N-1), 3)) # change last "3" to 1 for N=2
#end
#graph_lst = $_graph_lst

@with_kw mutable struct LeakyIntegratorPars
	K
	R
	L_inv
	C_inv
	v_ref
	n_prod
	n_cons
end

@with_kw mutable struct ILCPars
	kappa
	mismatch_yesterday
	daily_background_power
	current_background_power
	ilc_nodes
	ilc_covers
	Q
end
 @with_kw mutable struct incidences
	 inc_i
	 inc_v
end

@with_kw mutable struct UlMoparss
	N::Int
	ll::LeakyIntegratorPars
	hl::ILCPars
	inc::incidences
	periodic_infeed
	periodic_demand
	fluctuating_infeed
	residual_demand
	incidence

	function UlMoparss(N::Int,
						ll::LeakyIntegratorPars,
						hl:: ILCPars,
						inc::incidences,
						periodic_infeed,
						periodic_demand,
						fluctuating_infeed,
						residual_demand)
			new(N, ll,
			hl,
			inc,
			periodic_infeed,
			periodic_demand,
			fluctuating_infeed,
			residual_demand,
			incidence_matrix(graph,oriented=true))# incidence matrix vielleicht überprüfen , müssen zu 171 passen
	end
end

function set_parameters(N, kappa, Q)
	low_layer_control = LeakyIntegratorPars(K = [0.1, 1., 0.1, 1.] , R = 0.0532, L_inv = 1/0.237e-4, C_inv = 1/0.01,  v_ref = 48. ,  n_prod = num_prod, n_cons = N.-num_prod)
	control_incidences = incidences(inc_i = zeros(N), inc_v = zeros(Int(1.5*N)))
	higher_layer_control = ILCPars(kappa = kappa, mismatch_yesterday=zeros(24,N), daily_background_power=zeros(24,N), current_background_power=zeros(N), ilc_nodes=1:N, ilc_covers = [], Q = Q)
	periodic_infeed = t -> zeros(N)
	peak_demand = rand(N)
	periodic_demand = t -> zeros(N)
	fluctuating_infeed = t -> zeros(N)
	residual_demand = t -> zeros(N)

	return UlMoparss(N,low_layer_control,
					higher_layer_control,
					control_incidences,
					periodic_infeed,
					periodic_demand,
					fluctuating_infeed,
					residual_demand)
end
############################################
function prosumerToymodel!(du, u, p, t)

	n_lines = Int(1.5*p.N)

    #state variables
    i = u[1:n_lines]
    v = u[(n_lines+1):Int(2.5*p.N)]

    di = @view du[1:n_lines]
    dv = @view du[(n_lines+1):Int(2.5*p.N)]
    control_power_integrator = @view du[Int(2.5*p.N)+1:Int(3.5*p.N)]

	periodic_power =  p.periodic_demand(t) .+ p.periodic_infeed(t) #determine the update cycle of the hlc
	fluctuating_power =   p.residual_demand(t) .+ p.fluctuating_infeed(t) # here we can add fluctuating infeed as well
	Pd = periodic_power + fluctuating_power

    i_ILC =  p.hl.current_background_power./v

	p.inc.inc_v = p.incidence' * v
    p.inc.inc_i = p.incidence * i

	i_gen = p.ll.K .* (p.ll.v_ref .- v)
	i_load = Pd./(v.+1)

	@. di = p.ll.L_inv .*(-(p.ll.R.*i) .+ p.inc.inc_v)
	@. dv = p.ll.C_inv.*(i_ILC.+i_gen.- p.inc.inc_i .- i_load)

	@. control_power_integrator =  i_gen.* v 	#Power LI


	return nothing
end
@doc """
    HourlyUpdate()
Store the integrated control power in memory.
See also [`(hu::HourlyUpdate)`](@ref).
"""
struct HourlyUpdate
	integrated_control_power_history
	HourlyUpdate() = new([])
end



@doc """
    HourlyUpdate(integrator)
PeriodicCallback function acting on the `integrator` that is called every simulation hour (t = 1,2,3...).
"""
function (hu::HourlyUpdate)(integrator)
	hour = mod(round(Int, integrator.t/3600.), 24) + 1
	last_hour = mod(hour-2, 24) + 1
	power_idx = Int(2.5*integrator.p.N)+1:Int(3.5*integrator.p.N) # power index

	#power calculation y^c
	integrator.p.hl.mismatch_yesterday[last_hour,:] .= 1/3600 .* integrator.u[power_idx]
	integrator.u[power_idx] .= 0.

	integrator.p.hl.current_background_power .= integrator.p.hl.daily_background_power[hour, :]

	nothing
end



function DailyUpdate_X(integrator)
#ilc
	integrator.p.hl.daily_background_power =  integrator.p.hl.Q * (integrator.p.hl.daily_background_power + integrator.p.hl.kappa .* integrator.p.hl.mismatch_yesterday) #mismatch is horuly energy
	nothing
end
# Monte Carlo functions

get_run(i, batch_size) = mod(i, batch_size)==0 ? batch_size : mod(i, batch_size)
get_batch(i, batch_size) = 1 + (i - 1) ÷ batch_size


function prob_func_ic(prob, i, repeat, batch_size, kappa_lst, num_days)
	println("sim ", i)
	run = get_run(i, batch_size)
	batch = get_batch(i, batch_size)

	prob.p.hl.daily_background_power .= 0.
	prob.p.hl.current_background_power .= 0.
	prob.p.hl.mismatch_yesterday .= 0.

	prob.p.hl.kappa = kappa_lst[batch]

	#prob.p.coupling = 800. .* diagm(0=>ones(ne(prob.p.graph)))

	hourly_update = HourlyUpdate()

	ODEProblem(prosumerToymodel!, prob.u0, prob.tspan, prob.p,
		callback=CallbackSet(PeriodicCallback(hourly_update, 3600),
							 PeriodicCallback(DailyUpdate_X, 3600*24)))
end



function observer_ic(sol, i, energy_filter, num_days,N) # what should be extracted from one run
	# sol.prob.callback.discrete_callbacks[1].affect!.f.integrated_control_power_history

	hourly_energy = zeros(24*num_days,N)
	for i=1:24*num_days
		for j = 1:N
			hourly_energy[i,j] = sol(i*3600)[energy_filter[j]]./3600
		end
	end

	ILC_power = zeros(num_days,24,N)
	norm_energy_d = zeros(num_days,N)
	for j = 1:N
		norm_energy_d[1,j] = norm(hourly_energy[1:24,j])
	end

	for i=2:num_days
		for j = 1:N
			ILC_power[i,:,j] = sol.prob.p.hl.Q*(ILC_power[i-1,:,j] +  sol.prob.p.hl.kappa*hourly_energy[(i-1)*24+1:i*24,j])
		end
		for j = 1:N
			norm_energy_d[i,j] = norm(hourly_energy[(i-1)*24+1:i*24,j])
		end
	end

	((sol.prob.p.hl.kappa, hourly_energy, norm_energy_d), false)
end


############################################
# this should only run on one process
############################################

struct demand_amp_var
	demand
end


function (dav::demand_amp_var)(t)
	index = Int(floor(t / (24*3600)))
	dav.demand[index + 1,:]
end



demand_amp1 = demand_amp_var(60 .+ rand(num_days+1,Int(N/4)).* 40.)
demand_amp2 = demand_amp_var(70 .+ rand(num_days+1,Int(N/4)).* 30.)
demand_amp3 = demand_amp_var(80 .+ rand(num_days+1,Int(N/4)).* 20.)
demand_amp4 = demand_amp_var(90 .+ rand(num_days+1,Int(N/4)).* 10.)
demand_amp = t->vcat(demand_amp1(t), demand_amp2(t),demand_amp3(t),demand_amp4(t))



periodic_demand =  t-> demand_amp(t)./100 .* sin(t*pi/(24*3600))^2
samples = 24*4
inter = interpolate([.2 * randn(N) for i in 1:(num_days * samples + 1)], BSpline(Linear()))
residual_demand = t -> inter(1. + t / (24*3600) * samples) # 1. + is needed to avoid trying to access out of range

#########################################
#            SIM                     #
#########################################

##################  set higher_layer_control ##################

kappa = 1.
vc1 = 1:N # ilc_nodes (here: without communication)
cover1 = Dict([v => [] for v in vc1])# ilc_cover
u = [zeros(1000,1);1;zeros(1000,1)];
fc = 1/6;
a = digitalfilter(Lowpass(fc),Butterworth(2));
Q1 = filtfilt(a,u);# Markov Parameter
Q = Toeplitz(Q1[1001:1001+24-1],Q1[1001:1001+24-1]);


# kappa_lst = (0:0.01:2) ./ l_hour
begin
	kappa_lst = (0:.25:2)
	kappa = kappa_lst[1]
	num_monte = batch_size*length(kappa_lst)
end

################### set parameters ############################
begin
	param = set_parameters(N, kappa, Q)
	param.periodic_demand = periodic_demand
	param.residual_demand = residual_demand
	param.hl.daily_background_power .= 0
	param.hl.current_background_power .= 0
	param.hl.mismatch_yesterday .= 0.
end

begin
	fp = [0. 0. 0. 0. 0. 0. 48. 48. 48. 48. 0. 0. 0. 0.] #initial condition
	factor = 0.
	ic = factor .* ones(14)
	tspan = (0. , num_days * l_day) # 1 Tag
	tspan2 = (0., 10.0)
	#tspan3 = (0., 200.)
	ode = ODEProblem(prosumerToymodel!, fp, tspan, param,
	callback=CallbackSet(PeriodicCallback(HourlyUpdate(), l_hour),
						 PeriodicCallback(DailyUpdate_X, l_day)))
end
sol = solve(ode, Rodas4())

#################################### mein code kommt bis hier ##################
monte_prob = EnsembleProblem(
	ode,
	output_func = (sol, i) -> observer_ic(sol, i, energy_filter,num_days,N),
	prob_func = (prob,i,repeat) -> prob_func_ic(prob,i,repeat, batch_size, kappa_lst, num_days),
#	reduction = (u, data, I) -> experiments.reduction_ic(u, data, I, batch_size),
	u_init = [])

res = solve(monte_prob,
					 Rodas4P(),
					 trajectories=num_monte,
					 batch_size=batch_size)

kappa = [p[1] for p in res.u]
hourly_energy = [p[2] for p in res.u]
norm_energy_d = [p[3] for p in res.u]
###################################### ab hier unverändert ###############################################################
using LaTeXStrings
plot(mean(norm_energy_d[1],dims=2),legend=:topright, label = L"\kappa = 0\, h^{-1}", ytickfontsize=14,
               xtickfontsize=14, linestyle=:dot, margin=8Plots.mm,
    		   legendfontsize=8, linewidth=3,xaxis=("days [c]",font(14)), yaxis = ("2-norm of the error",font(14)), left_margin=12Plots.mm) #  ylims=(0,1e6)
plot!(mean(norm_energy_d[2],dims=2), label= L"\kappa = 0.25\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
plot!(mean(norm_energy_d[3],dims=2), label= L"\kappa = 0.5\, h^{-1}", linewidth = 3, linestyle=:dashdot)
plot!(mean(norm_energy_d[4],dims=2),label=  L"\kappa = 0.75\, h^{-1}", linewidth = 3, linestyle=:dash)
plot!(mean(norm_energy_d[5],dims=2), label= L"\kappa = 1\, h^{-1}", linewidth = 3, linestyle=:solid)
title!("Error norm")
savefig("$dir/plots/variation_kappa_leq_1_hetero.png")

using LaTeXStrings
plot(mean(norm_energy_d[5],dims=2),legend=:topright, label = L"\kappa = 1\, h^{-1}", ytickfontsize=14,
               xtickfontsize=14, linestyle =:solid, margin=8Plots.mm,left_margin=12Plots.mm,
    		   legendfontsize=8, linewidth=3,xaxis=("days [c]",font(14)), yaxis=("2-norm of the error",font(14)))  # ylims=(0,1e6)
plot!(mean(norm_energy_d[6],dims=2),label=  L"\kappa = 1.25\, h^{-1}", linewidth = 3, linestyle=:dash)
plot!(mean(norm_energy_d[7],dims=2),label=  L"\kappa = 1.5\, h^{-1}", linewidth = 3, linestyle=:dashdot)
plot!(mean(norm_energy_d[8],dims=2),label=  L"\kappa = 1.75\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
plot!(mean(norm_energy_d[9],dims=2), label= L"\kappa = 2 h^{-1}", linewidth = 3, linestyle=:dot)
#title!("Error norm")
savefig("$dir/plots/variation_kappa_geq_1_hetero.png")



# # never save the solutions INSIDE the git repo, they are too large, please make a folder solutions at the same level as the git repo and save them there
# jldopen("../../solutions/sol_def_N4.jld2", true, true, true, IOStream) do file
# 	file["sol1"] = sol1
# end
#
# @save "../../solutions/sol_kp525_ki0005_N4_pn_de-in_Q.jld2" sol1
