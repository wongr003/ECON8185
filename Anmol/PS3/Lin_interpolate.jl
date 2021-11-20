# This function performs linear interpolation of function f at point x ∈ [x0, x1] where f(x0) = y0 and f(x1) = y1
# Inputs: y0, y1, x0, x1, x
# Output: f(x) = y

function Lin_interpolate(y0, y1, x0, x1, x)
    y = (y0*(x1-x) + y1*(x-x0))/(x1-x0);
    return y
end