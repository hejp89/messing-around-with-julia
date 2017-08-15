using PyPlot

function logistic(t)
    u = e^(-t)
    return 1 / (1 + u)
end

function logistic_dervitive(t)
    u = e^(-t)
    return u / (1 + u)^2
end

# Determine the Jacobian of the logistic function i.e. f(x, b) = 1 / (1 + e^(b0 + b1 * x1 + ... + bn * xn)).
function J!(x, beta; recycle_matrix = nothing)
    result = recycle_matrix
    if result == nothing
        result = zeros(size(x)[1], size(beta)[1])
    end

    for i = 1:size(x)[1]
        l_dash = logistic_dervitive(dot(x[i, :], beta))

        for j = 1:size(x)[2]
            result[i, j] = x[i, j] * l_dash
        end
    end

    return result
end

# Calculate the vector of residuals observed (y) - predicted (logistic(x, beta)).
function r!(x, beta, y; recycle_matrix = nothing)
    result = recycle_matrix
    if result == nothing
        result = zeros(size(x)[1], 1)
    end

    for i = 1:size(x)[1]
        l = logistic(dot(x[i, :], beta))

        result[i, 1] = y[i] - l
    end

    return result
end

function sum_of_squares(r)
    return sum(dot(r, r))
end

# Calculate the optimial beta parameters for the supplied data.

# x - an array of floats with one or more columns.
# y - a single column array of 1s or 0s. Multiclass regression is not currently implemented. A one v rest model could be implemented by
#     combining multiple models.
function fit_logistic_model!(x, y; beta_initial = nothing, max_iterations = 10, res_delta_threshold = 0.01, print_residual = false)
    X = hcat(ones(size(x)[1]), x)

    beta = beta_initial
    if beta == nothing
        beta = zeros(size(X)[2], 1)
    end

    r_matrix = zeros(size(x)[1], 1)
    J_matrix = zeros(size(x)[1], size(beta)[1])

    current_res = sum_of_squares(r!(X, beta, y, recycle_matrix = r_matrix))
    if print_residual
        println("Residual Initial: $(current_res)")
    end

    for i = 1:max_iterations
        j = J!(X, beta, recycle_matrix = J_matrix)

        beta += inv(transpose(j) * j) * transpose(j) * r!(X, beta, y, recycle_matrix = r_matrix)
        # beta += pinv(j) * r(x, beta, y)

        res = sum_of_squares(r!(X, beta, y, recycle_matrix = r_matrix))

        if print_residual
            println("Residual Iteration $(i): $(res)")
        end

        if current_res - res < res_delta_threshold
            break
        end

        current_res = res
    end

    return beta
end

function predict(x, beta)
    X = hcat(ones(size(x)[1]), x)
    result = zeros(size(x)[1], 1)

    for i = 1:size(x)[1]
        result[i, 1] = logistic(dot(X[i, :], beta))
    end

    return result
end

# Example and plot of results

x = vcat(0.8 * randn(100) - 1, 0.8 * randn(100) + 1) # e.g. height
y = vcat(zeros(100), ones(100)) # e.g. gender

params = fit_logistic_model!(x, y, print_residual = true, res_delta_threshold = 0.00001)

x_prediction = linspace(floor(minimum(x)), ceil(maximum(x)), 100)
y_prediction = predict(x_prediction, params)

scatter(x, y, color="#4FC3F7", s=10)
plot(x_prediction, y_prediction, color="#212121")
