function [dldX, dldA, dldb] = fully_connected_backward(X, dldY, A, b)
    % Inputs
    %   X - The input variable. The size might vary, but the last dimension
    %       tells you which element in the batch it is.
    %   dldy - The partial derivatives of the loss with respect to the
    %       output varible y. The size of dldy is the same as for y as
    %       computed in the forward pass.
    %   A  - The weight matrix
    %   b  - The bias vector
    %
    % Outputs
    %    dldX - Gradient backpropagated to X
    %    dldA - Gradient backpropagated to A
    %    dldb - Gradient backpropagated to b
    %
    % All gradients should have the same size as the variable. That is,
    % dldx and x should have the same size, dldA and A the same size and dldb
    % and b also the same size.
    sz = size(X);
    batch = sz(end);
    features = prod(sz(1:end-1));

    % We reshape the input vector so that all features for a single batch
    % element are in the columns. X is now as defined in the assignment.
    X = reshape(X, [features, batch]);
    
    assert(size(A, 2) == features, ...
        sprintf('Expected %d columns in the weights matrix, got %d', features, size(A,2)));
    assert(size(A, 1) == numel(b), 'Expected as many rows in A as elements in b');
    
    % Implement it here.
    %error('Implement the backward pass!');
    dldX = A'*dldY;
    dldA = dldY*X';
    dldb = sum(dldY,2);
    
    % note that dldX should have the same size as X, so use reshape
    dldX = reshape(dldX, sz);
end
