
# A naive implementation of the calculation of a determinant of a matrix
function det_hp(M)
  matrix_size = size(M)

  if matrix_size[1] != matrix_size[2]
    throw(Exception("Matrix must be square"))
  end

  if matrix_size[1] == 2
    return M[1, 1] * M[2, 2] - M[2, 1] * M[1, 2]
  end

  if matrix_size[1] == 1
    return M[1, 1]
  end

  result = 0

  for row = 1:matrix_size[1]
    result += (-1) ^ (row + 1) * M[row, 1] * det_hp(slicedim(slicedim(M, 1, collect(i for i in 1:matrix_size[1] if i != row)), 2, 2:matrix_size[2]))
  end

  return result
end

# correctness test
no_of_tests = 1000
count_correct = 0
for i = 1:no_of_tests
    n = rand(2:10)
    m = reshape(collect(rand() for x in 1:n*n), n, n)
    if abs(det(m) - det(m)) < 1e-10
      count_correct += 1
    end
end

println("Correct tests $(count_correct) out of $(no_of_tests)\n")

# Performance test
n = 5
m = reshape(collect(rand() for x in 1:n*n), n, n)

println("Performance comparison\n")

println("det:")
tic()
for i = 1:no_of_tests
    det(m)
end
toc()

println("\ndet_hp:")
tic()
for i = 1:no_of_tests
    det_hp(m)
end
toc()
