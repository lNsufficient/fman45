function dldx = relu_backward(x, dldy)
    dldx = dldy.*(x>0);
end
