function logistic(t)
    u = e^(-t)
    return 1 / (1 + u)
end

function logistic_dervitive(t)
    u = e^(-t)
    return u / (1 + u)^2
end

# Determine the Jacobian of the logistic function i.e. f(x, b) = 1 / (1 + e^(b0 + b1 * x1 + ... + bn * xn))
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

#
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

function fit_logistic_model!(x, y; beta_initial = nothing, max_iterations = 10, res_delta_threshold = 0.01, print_residual = false)
    X = hcat(ones(size(x)[1]), x)

    beta = beta_initial
    if beta == nothing
        beta = zeros(size(X)[2], 1)
    end

    r_matrix = zeros(size(x)[1], 1)
    J_matrix = zeros(size(x)[1], size(beta)[1])

    current_res = sum_of_squares(r!(X, beta, y, recycle_matrix = r_matrix))
    println("Residual Initial: $(current_res)")

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

x = [0.92, 0.72, -0.08, -0.86, -0.74, -0.48, -0.31, 0.33, 0.21, 0.87, 0.83, -0.34, 0.77, -0.21, -0.86, -0.64, -0.69, -0.05, -0.89, 0.67,
     -0.11, -0.78, 0.14, -0.24, -0.39, -0.56, -0.13, 0.51, 0.36, -0.22, -0.22, -0.66, 0.67, -0.74, 0.37, -0.28, -0.99, -0.5, 0.71, -0.06,
     -0.5,  -0.54, -0.93, -0.5, 0.12, 0.91, 0.38, -0.97, -0.1, 0.21, -0.69, 0.47, 0.62, -0.77, 0.44, -0.5, -0.87, -0.24, 0.5, -0.78, 0.76,
     0.47, -0.4, -0.96, 0.33,  -0.88, -0.4, 0.63, 0.32, -0.7, -0.9, -0.13, 0.37, 0.7, -0.44, -0.31, 0.78, 0.58, 0.46, 0.6, 0.32, -0.1,
     -0.99, 0.65, -0.65, 0.36, 0.24, -0.05, -0.21, 0.16, 0.42, 0.98, -0.09, 0.62, -0.28, -0.08, 0.38, -0.67, 0.14, -0.37]

y = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,  0.0, 0.0,
     0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
     1.0,  0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,  0.0, 0.0,  0.0,  0.0,  1.0,  1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

tic()

params = fit_logistic_model!(x, y, print_residual = true, res_delta_threshold = 0.00001)

toc()
