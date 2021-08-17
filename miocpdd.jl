################################################################################
#                                                                              #
#  Julia code for the paper                                                    #
#  "Time-Domain Decomposition for Mixed-Integer Optimal Control Problems"      #
#  by Falk M. Hante, Richard Krug, Martin Schmidt                              #
#                                                                              #
#  Author: Richard Krug                                                        #
#                                                                              #
################################################################################

using DelimitedFiles
import GAMS
using JuMP
import MathOptInterface
using Plots
using Printf

plot_iterations = true
compute_obj_val = true
plot_result = true
plot_err = false

threshold_x = 1e-2
threshold_λ = 1e-2

solver_time = 1000 # s
overall_time = 1000 # s

number_of_domains = 4

γ = 1
ε = 0.5

number_of_time_steps = 100

solver = "antigone"

# discretization = "implEuler"
discretization = "explRK4"
# discretization = "RadauIIA"

test_number = 1

struct MIOCP
	A::Array{Float64,2}
	B::Array{Float64,2}
	c::Array{Float64,1}
	R₀::Array{Float64,2}
	c₀::Array{Float64,1}
	Rₜ::Array{Float64,2}
	cₜ::Array{Float64,1}
	Q₀::Array{Float64,2}
	q₀::Array{Float64,1}
	Qₜ::Array{Float64,2}
	qₜ::Array{Float64,1}
	Lᵤ::Array{Float64,2}
	u₋::Array{Float64,1}
	u₊::Array{Float64,1}
	NLrhs # Nonelinear ODE contraints. Overrides A, B, and c
	Constr # Additional control contraints
end

function get_vcp(miocp::MIOCP, xₖ::Array{Float64,2}, uₖ::Array{Float64,2}, Δt::Real, k::Int, K::Int, fix_u::Bool=false)
	n = size(xₖ, 1)
	m = size(uₖ, 1)
	number_of_steps = size(xₖ, 2) - 1

	model = Model(GAMS.Optimizer)
	x = @variable(model, x[i=1:n, j=1:(number_of_steps+1)], start=xₖ[i,j])
	if fix_u
		u = @variable(model, uₖ[i,j] <= u[i=1:m, j=1:number_of_steps] <= uₖ[i,j], Int, start=uₖ[i,j])
	else
		u = @variable(model, miocp.u₋[i] <= u[i=1:m, j=1:number_of_steps] <= miocp.u₊[i], Int, start=uₖ[i,j])
	end

	if !isnothing(miocp.Constr)
		miocp.Constr(model, x, u)
	end

	if isnothing(miocp.NLrhs)
		if discretization == "implEuler"
			@constraint(model, ode[step = 1:number_of_steps], (x[1:n,step+1] - x[1:n,step]) / Δt .== miocp.A * x[1:n,step+1] + miocp.B * u[1:m,step] + miocp.c)
		elseif is_discretization_implicit()
			a, b = get_butcher_tableau()
			number_of_stages = length(b)
			stages = @variable(model, stages[j=1:number_of_stages, i=1:n, step=1:number_of_steps])
			@constraint(model, [j = 1:number_of_stages, step = 1:number_of_steps], stages[j, 1:n, step] .== miocp.A * (x[1:n, step] + Δt * sum(a[j, l] * stages[l, 1:n, step] for l in 1:number_of_stages)) + miocp.B * u[1:m,step] + miocp.c)
			@constraint(model, [step = 1:number_of_steps], (x[1:n,step+1] - x[1:n,step]) / Δt .== sum(b[j] * stages[j, 1:n, step] for j in 1:number_of_stages))
		else
			a, b = get_butcher_tableau()
			number_of_stages = length(b)
			stages = Array{Any, 1}(undef, number_of_stages)
			for j in 1:number_of_stages
				stages[j] = @expression(model, [step = 1:number_of_steps], miocp.A * (x[1:n, step] + Δt * sum(a[j, l] * stages[l][step] for l in 1:(j-1))) + miocp.B * u[1:m,step] + miocp.c)
			end
			@constraint(model, [step = 1:number_of_steps], (x[1:n,step+1] - x[1:n,step]) / Δt .== sum(b[j] * stages[j][step] for j in 1:number_of_stages))
		end
	else
		if discretization == "implEuler"
			rhs = miocp.NLrhs(model, x[:, 2:end], u)
			for i in 1:n
				@NLconstraint(model, [step = 1:number_of_steps], (x[i,step+1] - x[i,step]) / Δt == rhs[i][step])
			end
		elseif is_discretization_implicit()
			a, b = get_butcher_tableau()
			number_of_stages = length(b)
			stages = @variable(model, stages[j=1:number_of_stages, i=1:n, step=1:number_of_steps])
			for j in 1:number_of_stages
				x_tmp = @NLexpression(model, [i = 1:n, step = 1:number_of_steps], x[i, step] + Δt * sum(a[j, l] * stages[l, i, step] for l in 1:number_of_stages))
				rhs = miocp.NLrhs(model, x_tmp, u)
				@NLconstraint(model, [i = 1:n, step = 1:number_of_steps], stages[j, i, step] == rhs[i][step])
			end
			for i in 1:n
				@NLconstraint(model, [step = 1:number_of_steps], (x[i,step+1] - x[i,step]) / Δt == sum(b[j] * stages[j, i, step] for j in 1:number_of_stages))
			end
		else
			a, b = get_butcher_tableau()
			number_of_stages = length(b)
			stages = Array{Any, 1}(undef, number_of_stages)
			for j in 1:number_of_stages
				x_tmp = @NLexpression(model, [i = 1:n, step = 1:number_of_steps], x[i, step] + Δt * sum(a[j, l] * stages[l][i][step] for l in 1:(j-1)))
				stages[j] = miocp.NLrhs(model, x_tmp, u)
			end
			for i in 1:n
				@NLconstraint(model, [step = 1:number_of_steps], (x[i,step+1] - x[i,step]) / Δt == sum(b[j] * stages[j][i][step] for j in 1:number_of_stages))
			end
		end
	end

	if k == 0
		@constraint(model, initial, miocp.R₀ * x[1:n,1] .== miocp.c₀)
	end
	if k == K
		@constraint(model, terminal, miocp.Rₜ * x[1:n, number_of_steps+1] .== miocp.cₜ)
	end
	# print(model, "\n")
	set_optimizer_attribute(model, "Solver", solver)
	set_optimizer_attribute(model, "LogOption", 0)
	# set_optimizer_attribute(model, "logOption", 4)
	set_optimizer_attribute(model, "ResLim", solver_time) # Time limit in seconds
	return model
end

function solve_vcp(model::Model, miocp::MIOCP, γ::Real, ϕ::Array{Float64,3}, xₖ::Array{Float64,2}, uₖ::Array{Float64,2}, λ::Array{Float64,3}, Δt::Real, k::Int)
	K = size(ϕ, 2)
	n = size(xₖ, 1)
	m = size(uₖ, 1)
	number_of_steps = size(xₖ, 2) - 1
	x = model[:x]
	u = model[:u]
	if K == 0
		@objective(model, Min, sum(miocp.Q₀[i,j] * x[i,1] * x[j,1] for i in 1:n, j in 1:n) / 2 + transpose(miocp.q₀) * x[1:n,1] + sum(miocp.Qₜ[i,j] * x[i,number_of_steps+1] * x[j,number_of_steps+1] for i in 1:n, j in 1:n) / 2 + transpose(miocp.qₜ) * x[1:n,number_of_steps+1] + sum(miocp.Lᵤ[i,j] * u[i,step] * u[j,step] for i in 1:m, j in 1:m, step in 1:number_of_steps) * (Δt / 2))
	elseif k == 0
		@objective(model, Min, sum(miocp.Q₀[i,j] * x[i,1] * x[j,1] for i in 1:n, j in 1:n) / 2 + transpose(miocp.q₀) * x[1:n,1] + sum((x[i,number_of_steps+1] - ϕ[2,k+1,i])^2 for i in 1:n) / (2 * γ) + sum(miocp.Lᵤ[i,j] * u[i,step] * u[j,step] for i in 1:m, j in 1:m, step in 1:number_of_steps) * (Δt / 2))
	elseif k == K
		@objective(model, Min, sum((x[i,1] - ϕ[1,k,i])^2 for i in 1:n) / (2 * γ) + sum(miocp.Qₜ[i,j] * x[i,number_of_steps+1] * x[j,number_of_steps+1] for i in 1:n, j in 1:n) / 2 + transpose(miocp.qₜ) * x[1:n,number_of_steps+1] + sum(miocp.Lᵤ[i,j] * u[i,step] * u[j,step] for i in 1:m, j in 1:m, step in 1:number_of_steps) * (Δt / 2))
	else
		@objective(model, Min, sum((x[i,1] - ϕ[1,k,i])^2 for i in 1:n) / (2 * γ) + sum((x[i,number_of_steps+1] - ϕ[2,k+1,i])^2 for i in 1:n) / (2 * γ) + sum(miocp.Lᵤ[i,j] * u[i,step] * u[j,step] for i in 1:m, j in 1:m, step in 1:number_of_steps) * (Δt / 2))
	end
	if has_values(model)
		set_start_value.(all_variables(model), value.(all_variables(model)))
	end

	optimize!(model)
	if primal_status(model) != MathOptInterface.FEASIBLE_POINT
		println(primal_status(model))
		println(dual_status(model))
		println(termination_status(model))
		throw(ErrorException("Could not solve virtual control problem"))
	end

	xₖ[:,:] = value.(x[1:n,1:(number_of_steps+1)])
	uₖ[:,:] = value.(u[1:m,1:number_of_steps])
	if k != 0
		λ[1,k,:] = (xₖ[:,1] - ϕ[1,k,:]) / γ
	end
	if k != K
		λ[2,k+1,:] = -(xₖ[:,number_of_steps+1] - ϕ[2,k+1,:]) / γ
	end
	return objective_value(model)
end

function add_iteration_errors(x_errors::Array{Array{Float64,2},1}, λ_errors::Array{Array{Float64,2},1}, x::Array{Array{Float64,2},1}, λ::Array{Float64,3})
	K = size(λ, 2)
	n = size(λ, 3)
	x_error = zeros(Float64, K, n)
	for k in 1:K
		x_error[k,:] = abs.(x[k][:,end] - x[k+1][:,1])
	end
	λ_error = abs.(λ[2,:,:] - λ[1,:,:])
	push!(x_errors, x_error)
	push!(λ_errors, λ_error)
end

function get_max_errors(x::Array{Array{Float64,2},1}, λ::Array{Float64,3})
	K = size(λ, 2)
	max_error_x = 0
	for k in 1:K
		max_error_x = max(max_error_x, maximum(abs.(x[k][:,end] - x[k+1][:,1])))
	end
	max_error_λ = (K > 0) ? maximum(abs.(λ[2,:,:] - λ[1,:,:])) : 0
	return (max_error_x, max_error_λ)
end

function plot_errors(x_errors::Array{Array{Float64,2},1}, λ_errors::Array{Array{Float64,2},1})
	number_of_iterations = length(x_errors)
	K = size(x_errors[1], 1)
	if K == 0
		return
	end
	n = size(x_errors[1], 2)
	p_iter = 1:number_of_iterations

	p = plot(xlabel="Iteration", legend=:topright)
	p_x = map(maximum, x_errors)
	plot!(p, p_iter, p_x, label="x", seriescolor=1)
	p_λ = map(maximum, λ_errors)
	plot!(p, p_iter, p_λ, label="λ", seriescolor=2)
	display(p)
end

function plot_iteration(ts::Array{<:Real,1}, Δt::Real, x::Array{Array{Float64,2},1}, u::Array{Array{Float64,2},1})
	n = size(x[1], 1)
	m = size(u[1], 1)
	K = length(ts) - 2
	p = plot(legend=:right)
	for k in 0:K
		number_of_steps = size(x[k+1], 2) - 1
		p_t = [ts[k+1] + Δt * s for s in 0:number_of_steps]
		for i in 1:n
			p_x = x[k+1][i,:]
			label = (k == 0) ? string("x", i) : ""
			plot!(p, p_t, p_x, label=label, seriescolor=i)
		end
	end
	for k in 0:K
		number_of_steps = size(u[k+1], 2)
		p_t = [ts[k+1] + Δt * s for s in 0:number_of_steps]
		for i in 1:m
			p_u = zeros(number_of_steps + 1)
			p_u[1:number_of_steps] = u[k+1][i,:]
			p_u[number_of_steps+1] = u[k+1][i,number_of_steps]
			label = (k == 0) ? (m == 1) ? "u" : string("u", i) : ""
			plot!(p, p_t, p_u, label=label, seriescolor=n+i, linetype=:steppost)
		end
	end
	display(p)
end

function algo(miocp::MIOCP, ts::Array{<:Real,1}, γ::Real, ε::Float64, Δt::Real, ϕ::Array{Float64,3}=Array{Float64,3}(undef, 0, 0, 0))
	# Init
	n = length(miocp.q₀)
	m = length(miocp.u₋)
	K = length(ts) - 2
	x = Array{Array{Float64,2},1}(undef, K+1)
	u = Array{Array{Float64,2},1}(undef, K+1)
	λ = Array{Float64,3}(undef, 2, K, n)
	x_errors = Array{Float64,2}[]
	λ_errors = Array{Float64,2}[]
	for k in 0:K
		tₖ = ts[k+1]
		tₖ₊₁ = ts[k+2]
		number_of_steps = ceil(Int, (tₖ₊₁ - tₖ) / Δt)
		ts[k+2] = tₖ + number_of_steps * Δt
		x[k+1] = zeros(Float64, n, number_of_steps+1)
		u[k+1] = zeros(Float64, m, number_of_steps)
	end
	if length(ϕ) == 0
		ϕ = zeros(Float64, 2, K, n)
	end
	models = Array{Model,1}(undef, K+1)
	for k in 0:K
		models[k+1] = get_vcp(miocp, x[k+1], u[k+1], Δt, k, K)
	end

	# Iterations
	println("")
	@printf("%5s %8s %10s %10s\n", "Iter", "Time", "Error x", "Error λ")
	iter = 1
	time_all = 0
	while true
		time = @elapsed begin
			Threads.@threads for k in 0:K
			# for k in 0:K
				solve_vcp(models[k+1], miocp, γ, ϕ, x[k+1], u[k+1], λ, Δt, k)
			end
			for k in 1:K
				ϕ[1,k,:] = (1 - ε) * (x[k][:,end] - γ * λ[2,k,:]) + ε * (x[k+1][:,1] - γ * λ[1,k,:])
				ϕ[2,k,:] = (1 - ε) * (x[k+1][:,1] + γ * λ[1,k,:]) + ε * (x[k][:,end] + γ * λ[2,k,:])
			end
			add_iteration_errors(x_errors, λ_errors, x, λ)
			max_error_x, max_error_λ = get_max_errors(x, λ)
		end
		@printf("%5.0f %8.3f %10.5f %10.5f\n", iter, time, max_error_x, max_error_λ)
		# Plot
		if plot_iterations
			plot_iteration(ts, Δt, x, u)
		end
		if (max_error_λ <= threshold_λ) && (max_error_x <= threshold_x)
			println("Continuous solution found!")
			println(string("Iterations: ", iter))
			return (x, u, x_errors, λ_errors)
		end
		iter += 1
		time_all += time
		if time_all >= overall_time
			throw(ErrorException("OVERALL TIME LIMIT"))
		end
	end
end

function get_objective_value(miocp::MIOCP, x::Array{Array{Float64,2},1}, u::Array{Array{Float64,2},1}, Δt::Real)
	K = length(x) - 1
	n = size(x[1], 1)
	obj_val = 0
	if compute_obj_val
		u_all = hcat(u...)
		number_of_steps = size(u_all, 2)
		x_all = zeros(Float64, n, number_of_steps+1)
		index = number_of_steps+1
		for k in K:-1:0
			s = size(x[k+1], 2)
			x_all[:, (index-s+1):index] = x[k+1]
			index -= s - 1
		end
		model_all = get_vcp(miocp, x_all, u_all, Δt, 0, 0, true)
		obj_val = solve_vcp(model_all, miocp, 1, Array{Float64,3}(undef, 0, 0, 0), x_all, u_all, Array{Float64,3}(undef, 0, 0, 0), Δt, 0)
		if test_number == 2 || test_number == 3
			obj_val += 0.01 * 0.01
		end
		@printf("Objective value: %12.6f\n", obj_val)
	end
	return x_all, u_all, obj_val
end

function is_discretization_implicit()
	if discretization in ["explRK4"]
		return false
	end
	return true
end

function get_butcher_tableau()
	if discretization == "explRK4"
		a = [0 0 0 0; 1//2 0 0 0; 0 1//2 0 0; 0 0 1 0]
		b = [1//6 1//3 1//3 1//6]
	elseif discretization == "RadauIIA"
		a = [5//12 -1//12; 3//4 1//4]
		b = [3//4 1//4]
	end
	return (a, b)
end

function save_data_file(filename::String, t0::Real, Δt::Real, x::Array{Float64,2}, u::Array{Float64,2})
	n = size(x, 1)
	m = size(u, 1)
	number_of_steps = size(x, 2) - 1
	data = Array{Any,2}(undef, number_of_steps+2, 1+n+m)

	data[1, 1] = "t"
	data[2:end, 1] = convert(Array{Float64}, [t0 + Δt * s for s in 0:number_of_steps])
	for i in 1:n
		data[1, 1+i] = string("x", i)
		data[2:end, 1+i] = x[i, :]
	end
	for i in 1:m
		data[1, 1+n+i] = string("u", i)
		data[2:end, 1+n+i] = [u[i, :]..., u[i, end]]
	end

	writedlm(filename * ".dat", data, " ")
end

function save_error_data_file(filename::String, x_errors::Array{Array{Float64,2},1}, λ_errors::Array{Array{Float64,2},1})
	number_of_iterations = length(x_errors)
	data = Array{Any,2}(undef, number_of_iterations+1, 3)

	data[1, 1] = "Iteration"
	data[2:end, 1] = 1:number_of_iterations
	data[1, 2] = "x"
	data[2:end, 2] = map(maximum, x_errors)
	data[1, 3] = "lambda"
	data[2:end, 3] = map(maximum, λ_errors)

	writedlm(filename * ".dat", data, " ")
end

function run_miocp(miocp::MIOCP, t_max, ϕ::Array{Float64,3}=Array{Float64,3}(undef, 0, 0, 0))
	try
		ts = collect(0:(t_max // number_of_domains):t_max)
		Δt = t_max // number_of_time_steps
		time = @elapsed begin
			x, u, x_errors, λ_errors = algo(miocp, ts, γ, ε, Δt, ϕ)
		end
		@printf("Iteration time: %12.3f\n", time)
		if compute_obj_val
			obj_time = @elapsed begin
				x_all, u_all, obj_val = get_objective_value(miocp, x, u, Δt)
			end
			if plot_result
				if test_number == 3
					u_all = u_all[1:1, :] / 3 + 2 * u_all[2:2, :] / 3 + u_all[3:3, :]
				end
				plot_iteration([ts[1], ts[end]], Δt, [x_all], [u_all])
				# filename = "example1_decomposed"
				# save_data_file(filename, ts[1], Δt, x_all, u_all)
				# savefig(filename * ".pdf")
			end
			@printf("Overall time: %12.3f\n", time + obj_time)
		end
		if plot_err
			plot_errors(x_errors, λ_errors)
			# filename = "example1_gamma1"
			# save_error_data_file(filename, x_errors, λ_errors)
			# savefig(filename * ".pdf")
		end
	catch e
		print(e)
		println("")
	end
end

# A Mixed-Integer Linear-Quadratic Problem
function test1()
	t_max = 1
	A = Float64[0 2; -1 1]
	B = hcat(Float64[0, -1])
	c = Float64[0, 0]
	R₀ = Float64[1 0; 0 1]
	c₀ = Float64[-2, 1]
	Rₜ = Float64[0 0; 0 0]
	cₜ = Float64[0, 0]
	Q₀ = Float64[0 0; 0 0]
	q₀ = Float64[0, 0]
	Qₜ = Float64[2 0; 0 2]
	qₜ = Float64[0, 0]
	Lᵤ = hcat(Float64[1e-2])
	u₋ = Float64[0]
	u₊ = Float64[4]
	NLrhs = nothing
	Constr = nothing
	miocp = MIOCP(A, B, c, R₀, c₀, Rₜ, cₜ, Q₀, q₀, Qₜ, qₜ, Lᵤ, u₋, u₊, NLrhs, Constr)
	run_miocp(miocp, t_max)
end

# Fuller's initial value problem
function test2()
	t_max = 1
	A = Float64[0 0; 0 0]
	B = hcat(Float64[0, 0])
	c = Float64[0, 0]
	R₀ = Float64[1 0 0; 0 1 0; 0 0 1]
	c₀ = Float64[0.01, 0, 0]
	Rₜ = Float64[0 0 0; 0 0 0; 0 0 0]
	cₜ = Float64[0, 0, 0]
	Q₀ = Float64[0 0 0; 0 0 0; 0 0 0]
	q₀ = Float64[0, 0, 0]
	Qₜ = Float64[2 0 0; 0 2 0; 0 0 0]
	qₜ = Float64[-0.02, 0, 1]
	Lᵤ = hcat(Float64[0])
	u₋ = Float64[0]
	u₊ = Float64[1]
	NLrhs(model, x, u) = begin
		n = 3
		number_of_steps = size(u, 2)
		rhs = Array{Any, 1}(undef, n)
		rhs[1] = @NLexpression(model, [step = 1:number_of_steps], x[2, step])
		rhs[2] = @NLexpression(model, [step = 1:number_of_steps], 1 - 2 * u[1, step])
		rhs[3] = @NLexpression(model, [step = 1:number_of_steps], x[1, step]^2)
		return rhs
	end
	Constr = nothing
	miocp = MIOCP(A, B, c, R₀, c₀, Rₜ, cₜ, Q₀, q₀, Qₜ, qₜ, Lᵤ, u₋, u₊, NLrhs, Constr)
	run_miocp(miocp, t_max)
end

# Fuller's initial value multimode problem
function test3()
	t_max = 1
	A = Float64[0 0; 0 0]
	B = hcat(Float64[0, 0])
	c = Float64[0, 0]
	R₀ = Float64[1 0 0; 0 1 0; 0 0 1]
	c₀ = Float64[0.01, 0, 0]
	Rₜ = Float64[0 0 0; 0 0 0; 0 0 0]
	cₜ = Float64[0, 0, 0]
	Q₀ = Float64[0 0 0; 0 0 0; 0 0 0]
	q₀ = Float64[0, 0, 0]
	Qₜ = Float64[2 0 0; 0 2 0; 0 0 0]
	qₜ = Float64[-0.02, 0, 1]
	Lᵤ = Float64[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]
	u₋ = Float64[0, 0, 0, 0]
	u₊ = Float64[1, 1, 1, 1]
	NLrhs(model, x, u) = begin
		n = 3
		number_of_steps = size(u, 2)
		rhs = Array{Any, 1}(undef, n)
		rhs[1] = @NLexpression(model, [step = 1:number_of_steps], x[2, step])
		rhs[2] = @NLexpression(model, [step = 1:number_of_steps], 1 - 2 * u[1, step] - 0.5 * u[2, step] - 3 * u[3, step])
		rhs[3] = @NLexpression(model, [step = 1:number_of_steps], x[1, step]^2)
		return rhs
	end
	Constr(model, x, u) = begin
		number_of_steps = size(u, 2)
		@constraint(model, [step = 1:number_of_steps], sum(u[i,step] for i in 1:4) == 1)
	end
	miocp = MIOCP(A, B, c, R₀, c₀, Rₜ, cₜ, Q₀, q₀, Qₜ, qₜ, Lᵤ, u₋, u₊, NLrhs, Constr)
	run_miocp(miocp, t_max)
end

# F-8 aircraft
function test4()
	t_max = 1
	A = Float64[0 0; 0 0]
	B = hcat(Float64[0, 0])
	c = Float64[0, 0]
	R₀ = Float64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0]
	c₀ = Float64[0.4655, 0, 0, 0]
	Rₜ = Float64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0]
	cₜ = Float64[0, 0, 0, 0]
	Q₀ = Float64[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]
	q₀ = Float64[0, 0, 0, 0]
	Qₜ = Float64[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 100]
	qₜ = Float64[0, 0, 0, 0]
	Lᵤ = hcat(Float64[0])
	u₋ = Float64[0]
	u₊ = Float64[1]
	NLrhs(model, x, u) = begin
		n = 4
		xi = 0.05236
		number_of_steps = size(u, 2)
		rhs = Array{Any, 1}(undef, n)
		rhs[1] = @NLexpression(model, [step = 1:number_of_steps], x[4,step] * (-0.877 * x[1,step] + x[3,step] - 0.088 * x[1,step] * x[3,step] + 0.47 * x[1,step]^2 - 0.019 * x[2,step]^2 - x[1,step]^2 * x[3,step] + 3.846 * x[1,step]^3 + 0.215 * xi - 0.28 * x[1,step]^2 * xi + 0.47 * x[1,step] * xi^2 - 0.63 * xi^3 - (0.215 * xi - 0.28 * x[1,step]^2 * xi - 0.63 * xi^3) * 2 * u[1,step]))
		rhs[2] = @NLexpression(model, [step = 1:number_of_steps], x[4,step] * x[3,step])
		rhs[3] = @NLexpression(model, [step = 1:number_of_steps], x[4,step] * (-4.208 * x[1,step] - 0.396 * x[3,step] - 0.47 * x[1,step]^2 - 3.564 * x[1,step]^3 + 20.967 * xi - 6.265 * x[1,step]^2 * xi + 46 * x[1,step] * xi^2 - 61.4 * xi^3 - (20.967 * xi - 6.265 * x[1,step]^2 * xi - 61.4 * xi^3) * 2 * u[1,step]))
		rhs[4] = @NLexpression(model, [step = 1:number_of_steps], 0)
		return rhs
	end
	Constr(model, x, u) = begin
		number_of_steps = size(u, 2)
		@constraint(model, [step = 1:(number_of_steps+1)], x[4,step] >= 0)
	end
	# Constr = nothing
	miocp = MIOCP(A, B, c, R₀, c₀, Rₜ, cₜ, Q₀, q₀, Qₜ, qₜ, Lᵤ, u₋, u₊, NLrhs, Constr)
	run_miocp(miocp, t_max)
end

function run_test()
	[test1, test2, test3, test4][test_number]()
end

# for nd in [1, 2, 4, 8, 16]
# 	global number_of_domains
# 	number_of_domains = nd
# 	println("")
# 	println(string("number_of_domains: ", number_of_domains))
# 	println("")
# 	run_test()
# end

# for epsilon in 0:0.1:0.9
# 	global ε
# 	println("")
# 	println(string("epsilon: ", epsilon))
# 	println("")
# 	ε = epsilon
# 	run_test()
# end

run_test()
