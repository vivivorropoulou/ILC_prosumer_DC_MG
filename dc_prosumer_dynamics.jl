begin
	using JLD2, FileIO, GraphIO, CSV, DataFrames
	using ForwardDiff
	using Distributed
	using LightGraphs # create network topologies
	using LinearAlgebra
	using DifferentialEquations
	using GraphPlot
	using Random
	using NetworkDynamics
	using Plots
	using Parameters
	using ToeplitzMatrices
	using DSP
	using LaTeXStrings
	using Distributions
	using StatsBase
	using Roots
	using Interpolations
	Random.seed!(42)
end

begin
	dir = @__DIR__
	N = 4 # Number of nodes
	N_half = Int(N/2)
	num_days = 7 # Number of days
	num_prod = 2
	num_cons = N.-num_prod
	l_day = 3600*24 # DemCurve.l_day
	l_hour = 3600 # DemCurve.l_hour
	l_minute = 60
	A = zeros(4,24)
	A[1,1] = 1

end


struct demand_amp_var
	demand
end


function (dav::demand_amp_var)(t)
	index = Int(floor(t / (24*3600)))
	dav.demand[index + 1,:]
end



begin
	graph = random_regular_graph(iseven(3N) ? N : (N-1), 3)

end


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
			incidence_matrix(graph,oriented=true))
	end
end

function set_parameters(N, kappa, Q)
	low_layer_control = LeakyIntegratorPars(K = 1., R = 0.0532, L_inv = 1/0.237e-4, C_inv = 1/0.01,  v_ref = 48. ,  n_prod = num_prod, n_cons = N.-num_prod) # Homogeneous scenario
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

begin
	current_filter = 1:Int(1.5N)
	voltage_filter = Int(1.5N)+1:Int(2.5N)
	energy_filter = Int(2.5N)+1:Int(3.5N)#3N+1:4N
end

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

    i_ILC =  p.hl.current_background_power./v # power ILC in form of a current

	p.inc.inc_v = p.incidence' * v # incidence matrices
    p.inc.inc_i = p.incidence * i

	u_Ll = p.ll.K .* (p.ll.v_ref .- v) # generated current i_gen
	i_load = (periodic_power .+ fluctuating_power)./(v.+1) # load current

	@. di = p.ll.L_inv .*(-(p.ll.R.*i) .+ p.inc.inc_v)
	@. dv = p.ll.C_inv.*(i_ILC.+u_Ll.- p.inc.inc_i .- i_load)

	@. control_power_integrator = u_Ll.* v 	#power LI


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


#neues daily Update

function DailyUpdate_X(integrator)
#ilc
	integrator.p.hl.daily_background_power =  integrator.p.hl.Q * (integrator.p.hl.daily_background_power + integrator.p.hl.kappa .* integrator.p.hl.mismatch_yesterday)
	#print(size(integrator.p.hl.daily_background_power))
	#print(size(integrator.p.hl.mismatch_yesterday))
	nothing
end

function DailyUpdate_XI(integrator)
	integrator.p.hl.daily_background_power = integrator.p.hl.Q * (integrator.p.hl.daily_background_power + integrator.p.hl.mismatch_yesterday * integrator.p.hl.kappa')
	nothing
end

demand_amp1 = demand_amp_var(repeat([80 80 80 10 10 10 40 40 40 40 40], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp2 = demand_amp_var(repeat([10 10 10 80 80 80 40 40 40 40 40], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp3 = demand_amp_var(repeat([60 60 60 60 10 10 10 40 40 40 40], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp4 = demand_amp_var(repeat([30 30 30 30 10 10 10 80 80 80 80], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp = t->vcat(demand_amp1(t), demand_amp2(t), demand_amp3(t), demand_amp4(t))


periodic_demand =  t-> demand_amp(t) .* sin(t*pi/(24*3600))^2

samples = 24*4

inter = interpolate([10. * randn(N) for i in 1:(num_days * samples + 1)], BSpline(Linear()))
residual_demand = t -> inter(1. + t / (24*3600) * samples)
#plot(0:num_days, t->residual_demand)
##################  set higher_layer_control ##################

kappa = 1.
kappa2 = diagm(0 => [1.0, 0., 0., 0.])
vc1 = 1:N # ilc_nodes (here: without communication)
cover1 = Dict([v => [] for v in vc1])# ilc_cover
u = [zeros(1000,1);1;zeros(1000,1)];
fc = 1/6;
a = digitalfilter(Lowpass(fc),Butterworth(2));
Q1 = filtfilt(a,u);# Markov Parameter
Q = Toeplitz(Q1[1001:1001+24-1],Q1[1001:1001+24-1]);

################### set parameters ############################
begin
	param = set_parameters(N, kappa, Q)
	param.periodic_demand = periodic_demand
	param.residual_demand = residual_demand
	param.hl.daily_background_power .= 0
	param.hl.current_background_power .= 0
	param.hl.mismatch_yesterday .= 0.
end
####################### solving ###############################
begin
	fp = [0. 0. 0. 0. 0. 0. 48. 48. 48. 48. 0. 0. 0. 0.] #initial condition
	factor = 0.
	ic = factor .* ones(14)
	tspan = (0. , num_days * l_day)
	tspan2 = (0., 10.0)
	#tspan3 = (0., 200.)
	ode = ODEProblem(prosumerToymodel!, fp, tspan, param,
	callback=CallbackSet(PeriodicCallback(HourlyUpdate(), l_hour),
						 PeriodicCallback(DailyUpdate_X, l_day)))
end
sol = solve(ode, Rodas4())


#################################################################################
######################## ENERGIES ########################################
hourly_energy = zeros(24*num_days+1,N)

for i=1:24*num_days+1
	for j = 1:N
		hourly_energy[i,j] = sol((i-1)*3600)[energy_filter[j]]./3600 #the hourly integrated low-level control energy
	end
end

ILC_power = zeros(num_days+2,24,N)
for j = 1:N
	ILC_power[2,:,j] = Q*(zeros(24,1) +  kappa*hourly_energy[1:24,j]) # the ILC power on every node
	#ILC_power[2,:,j] = Q*(zeros(24,1) +  kappa2[j,j]*hourly_energy[1:24,j]) #
end
norm_energy_d = zeros(num_days,N)
for j = 1:N
	norm_energy_d[1,j] = norm(hourly_energy[1:24,j])
end

for i=2:num_days
	for j = 1:N
		ILC_power[i+1,:,j] = Q*(ILC_power[i,:,j] +  kappa*hourly_energy[(i-1)*24+1:i*24,j])
		#ILC_power[i+1,:,j] = Q*(ILC_power[i,:,j] +  kappa2[j,j]*hourly_energy[(i-1)*24+1:i*24,j])
		norm_energy_d[i,j] = norm(hourly_energy[(i-1)*24+1:i*24,j])
	end
end

ILC_power_agg = [norm(mean(ILC_power,dims=3)[d,:]) for d in 1:num_days+2]
ILC_power_hourly_mean = vcat(mean(ILC_power,dims=3)[:,:,1]'...)
ILC_power_hourly_mean_node1 = vcat(ILC_power[:,:,1]'...)
ILC_power_hourly = [norm(reshape(ILC_power,(num_days+2)*24,N)[h,:]) for h in 1:24*(num_days+2)]
ILC_power_hourly_node1 = [norm(reshape(ILC_power,(num_days+2)*24,N)[h,1]) for h in 1:24*(num_days+2)]
dd = t->((-periodic_demand(t) .- residual_demand(t)))
per_dd = t->(-periodic_demand(t))
load_amp = [first(maximum(dd(t))) for t in 1:3600*24:3600*24*num_days]

norm_hourly_energy = [norm(hourly_energy[h,:]) for h in 1:24*num_days]

################################## PLOTTING ########################################

#

node = 1
p1 = plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
plot!(0:num_days*l_day, t -> -dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)
plot!(1:3600:24*num_days*3600,hourly_energy[1:num_days*24,node], label=latexstring("y_$node^{c,h}"),linewidth=3) #, linestyle=:dash)
plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_node[1:num_days*24], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=14,
    		   legendfontsize=10, linewidth=3, yaxis=("normed power",font(14)),legend=false, lc =:black, margin=5Plots.mm)
title!(latexstring("j = $(node), K_D = 1.0\\, \\Omega ^{-1}"))
#ylims!(-30,100)
ylims!(-25,120)
#savefig("$dir/plots/kappa_1/K_variance/K=[0.1,0.1,1,0.1]/DC_prosumer_demand_seconds_$(node)_hetero.png")
#savefig("$dir/plots/manual_calc_variation_kappa/kappa_1/K_variance/K=[0.1,1,2,5]/DC_prosumer_10_demand_seconds_$(node)_hetero.png")
savefig("$dir/plots/kappa_[1-0-0-0]/DC_prosumer_10_demand_seconds_$(node)_hetero.png")

node = 2
p2 = plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
plot!(0:num_days*l_day, t -> -dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)
plot!(1:3600:24*num_days*3600,hourly_energy[1:num_days*24,node], label=latexstring("y_$node^{c,h}"),linewidth=3) #, linestyle=:dash)
plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_node[1:num_days*24], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=14,
    		   legendfontsize=10, linewidth=3,legend=false, lc =:black, margin=5Plots.mm)
#savefig("$dir/plots/kappa_1/DC_prosumer_demand_seconds_node_$(node)_hetero.png")
title!(latexstring("j = $(node), K_D = 1.0\\, \\Omega ^{-1}"))
#ylims!(-30,100)
ylims!(-25,120)
#savefig("$dir/plots/kappa_1/K_variance/K=[0.1,0.1,1,0.1]/DC_prosumer_demand_seconds_$(node)_hetero.png")
#savefig("$dir/plots/manual_calc_variation_kappa/kappa_1/K_variance/K=[0.1,1,2,5]/DC_prosumer_10_demand_seconds_$(node)_hetero.png")
savefig("$dir/plots/kappa_[1-0-0-0]/DC_prosumer_10_demand_seconds_$(node)_hetero.png")

node = 3
p3 = plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
plot!(0:num_days*l_day, t -> -dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)
plot!(1:3600:24*num_days*3600,hourly_energy[1:num_days*24,node], label=latexstring("y_$node^{c,h}"),linewidth=3) #, linestyle=:dash)
plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_node[1:num_days*24], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=14,
    		   legendfontsize=10, linewidth=3, xaxis = ("days [c]",font(14)),yaxis=("normed power",font(14)),legend=false, lc =:black, margin=5Plots.mm)
#savefig("$dir/plots/kappa_1/DC_prosumer_demand_seconds_node_$(node)_hetero.png")
title!(latexstring("j = $(node), K_D = 1.0\\, \\Omega ^{-1}"))
#ylims!(-30,100)
ylims!(-25,120)
#savefig("$dir/plots/kappa_1/K_variance/K=[0.1,0.1,1,0.1]/DC_prosumer_demand_seconds_$(node)_hetero.png")
#savefig("$dir/plots/manual_calc_variation_kappa/kappa_1/K_variance/K=[0.1,1,2,5]/DC_prosumer_10_demand_seconds_$(node)_hetero.png")
savefig("$dir/plots/kappa_[1-0-0-0]/DC_prosumer_10_demand_seconds_$(node)_hetero.png")

node = 4
p4 = plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
plot!(0:num_days*l_day, t -> -dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)
plot!(1:3600:24*num_days*3600,hourly_energy[1:num_days*24,node], label=latexstring("y_$node^{c,h}"),linewidth=3) #, linestyle=:dash)
plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_node[1:num_days*24], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=14,
    		   legendfontsize=10, linewidth=3, xaxis = ("days [c]",font(14)),legend=false, lc =:black, margin=5Plots.mm)
#savefig("$dir/plots/kappa_1/DC_prosumer_demand_seconds_node_$(node)_hetero.png")
title!(latexstring("j = $(node), K_D = 1.0\\, \\Omega ^{-1}"))
#ylims!(-30,100)
ylims!(-25,120)
#savefig("$dir/plots/kappa_1/K_variance/K=[0.1,0.1,1,0.1]/DC_prosumer_demand_seconds_$(node)_hetero.png")
#savefig("$dir/plots/manual_calc_variation_kappa/kappa_1/K_variance/K=[0.1,1,2,5]/DC_prosumer_10_demand_seconds_$(node)_hetero.png")
savefig("$dir/plots/kappa_[1-0-0-0]/DC_prosumer_10_demand_seconds_$(node)_hetero.png")

# SUM
psum = plot()
ILC_power_hourly_mean_sum = (vcat(ILC_power[:,:,1]'...) .+ vcat(ILC_power[:,:,2]'...) .+ vcat(ILC_power[:,:,3]'...) .+ vcat(ILC_power[:,:,4]'...))
plot!(0:num_days*l_day, t -> -(dd(t)[1] .+ dd(t)[2] .+ dd(t)[3] .+ dd(t)[4]),label = "total demand",xticks = (0:3600*24:24*3600, string.(0:num_days)),linewidth=3)
#plot!(0:l_day, t -> -(per_dd(t)[1] .+ per_dd(t)[2] .+ per_dd(t)[3] .+ per_dd(t)[4]), xticks = (0:3600*24:24*3600, string.(0:num_days)),label = "baseline",linewidth=3)
plot!(1:3600:24*num_days*3600,(hourly_energy[1:num_days*24,1] + hourly_energy[1:num_days*24,2] + hourly_energy[1:num_days*24,3] + hourly_energy[1:num_days*24,4]), label=latexstring("y^{c,h}"),linewidth=3, linestyle=:dash)
plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_sum[1:num_days*24], label=latexstring("\$u_^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=14,
    		   legendfontsize=10, linewidth=3, xaxis = ("days [c]",font(14)),yaxis=("normed power",font(14)),legend=false, lc =:black, margin=5Plots.mm)
title!(latexstring("\\sum_j, K_{D,j} = 1.0\\, \\Omega^{-1}"))
#title!(latexstring("K_D = 1, \\kappa = 1"))
#savefig("$dir/plots/kappa_1/kappa_1_DC_prosumer_demand_seconds_sum_hetero.png")
#title!(latexstring("\\kappa = 2"))
#savefig("$dir/plots/kappa_1/Powers_K_[0.1_1_0.1_1]_node_sum_hetero.png")
#savefig("$dir/plots/manual_calc_variation_kappa/kappa_1/K=1/DC_prosumer_10_demand_seconds_sum_hetero.png")
savefig("$dir/plots/manual_calc_variation_kappa/kappa_1/K_variance/K=[0.1,1,2,5]/DC_prosumer_10_demand_seconds_sum_hetero.png")
savefig("$dir/plots/kappa_[1-0-0-0]/DC_prosumer_10_demand_seconds_sum_hetero.png")
p_dif_nodes = plot(p1,p2,p3,p4, legend=false)
#savefig("$dir/plots/kappa_1/K_variance/K_[0.1_1_0.1_1]_seperate.png")
#savefig("$dir/plots/manual_calc_variation_kappa/kappa_1/K=1/DC_prosumer_10_demand_seconds_nodes_hetero.png")
savefig("$dir/plots/manual_calc_variation_kappa/kappa_1/K_variance/K=[0.1,1,2,5]/DC_prosumer_10_demand_seconds_nodes_hetero.png")
savefig("$dir/plots/kappa_[1-0-0-0]/DC_prosumer_10_demand_seconds_nodes_hetero.png")


#################################################################################
##################### HOURLY ENERGY CURRENT AND VOLTAGE PLOTTING ##############################

cur = plot(sol, vars = current_filter, title = "Current per edge ", label = ["Edge 1" "Edge 2" "Edge 3" "Edge 4" "Edge 5" "Edge 6"])
xlabel!("Time in s")
ylabel!("Current in A")
savefig("$dir/plots/DC_prosumer_current_per_edge.png")

volt = plot(sol, vars = voltage_filter,title = "Voltage per node ")
xlabel!("Time in s")
ylabel!("Voltage in V")
savefig("$dir/plots/DC_prosumer_voltage_per_node.png")


ener = plot(sol, vars = energy_filter, title = "Energy per node", label = ["Node 1" "Node 2" "Node 3" "Node 4"])
xlabel!("Time in s")
ylabel!("Power in W")
savefig("$dir/plots/DC_prosumer_constant_power_no_ILC_voltage_per_node.png")


plot(hourly_energy,title = "hourly energy", label = ["Node 1" "Node 2" "Node 3" "Node 4"])
plot!(hourly_energy[:,1] .+ hourly_energy[:,2].+hourly_energy[:,3] .+hourly_energy[:,4],label = "sum nodes")

xlabel!("Time in h")
ylabel!("Power in W")
savefig("$dir/plots/kappa_1/energy_bilance.png")
savefig("$dir/plots/kappa_1/K_variance/K_[0.1_1_0.1_1]_hourly_energy.png")
savefig("$dir/plots/kappa_1/K_variance/K_[0.1_1_2_5]_hourly_energy.png")
hourly_current = zeros(24*num_days+1,Int(1.5*N))

for i=1:24*num_days+1
	for j = 1:Int(1.5*N)
		hourly_current[i,j] = sol((i-1)*3600)[current_filter[j]] # weil das integral  auch durch 3600 geteilt wird
	end
end
plot(1:3600:24*num_days*3600,hourly_current[1:num_days*24,1],title = "Current per edge ", xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)))
plot!(1:3600:24*num_days*3600,hourly_current[1:num_days*24,2],title = "Current per edge ", xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)))
plot!(1:3600:24*num_days*3600,hourly_current[1:num_days*24,3],title = "Current per edge ", xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)))
plot!(1:3600:24*num_days*3600,hourly_current[1:num_days*24,4],title = "Current per edge ", xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)))
plot!(1:3600:24*num_days*3600,hourly_current[1:num_days*24,5],title = "Current per edge ", xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)))
plot!(1:3600:24*num_days*3600,hourly_current[1:num_days*24,6],title = "Current per edge ", xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)))

xlabel!("Time in h")
ylabel!("Current in A")
savefig("$dir/plots/DC_prosumer_no_ILC_current_per_edge.png")

hourly_voltage = zeros(24*num_days+1,N)

for i=1:24*num_days+1
	for j = 1:N
		hourly_voltage[i,j] = sol((i-1)*3600)[voltage_filter[j]] # weil das integral  auch durch 3600 geteilt wird
	end
end


for i in 1:24
	display(hourly_voltage[i,1])
end

plot(1:3600:24*num_days*3600,hourly_voltage[1:num_days*24,1],legend=:bottomright,xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), label = L"v_{gn,1}(t)")
plot!(1:3600:24*num_days*3600,hourly_voltage[1:num_days*24,2],legend=:bottomright,xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), label = L"v_{gn,2}(t)")
plot!(1:3600:24*num_days*3600,hourly_voltage[1:num_days*24,3],legend=:bottomright,xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), label = L"v_{gn,3}(t)")
plot!(1:3600:24*num_days*3600,hourly_voltage[1:num_days*24,4],legend=:bottomright,xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), label = L"v_{gn,4}(t)")
title!(L"v_{gn,j}(t)")
xlabel!("days [c]")
ylabel!("voltage [V]")
savefig("$dir/plots/voltages/DC_prosumer_ILC_kappa_[1-0-0-0]_voltage_nodes.png")



min_day_1 = findmin(hourly_voltage[1:24,1])

max_day_2 = findmax(hourly_voltage[25:48,1])

max_day_3 = findmax(hourly_voltage[49:72,1])

max_day_4 = findmax(hourly_voltage[73:96,1])

max_day_5 = findmax(hourly_voltage[97:120,1])

max_day_6 = findmax(hourly_voltage[121:144,1])

max_day_7 = findmax(hourly_voltage[145:169,1])

xlabel!("Time in h")
ylabel!("Voltage in V")
savefig("$dir/plots/DC_prosumer_ILC_voltage_per_node.png")
################################################################################
demand_plot = plot()
plot!(0:l_day, t -> -(dd(t)[1] .+ dd(t)[2] .+ dd(t)[3] .+ dd(t)[4]),label = "total demand",xticks = (0:3600*24:24*3600, string.(0:num_days)),linewidth=3)
plot!(0:l_day, t -> -(per_dd(t)[1] .+ per_dd(t)[2] .+ per_dd(t)[3] .+ per_dd(t)[4]), xticks = (0:3600*24:24*3600, string.(0:num_days)),label = "baseline",linewidth=3)
ylabel!("power demand [W]")
xlabel!("days")
savefig("$dir/plots/exemplary_demand.png")
