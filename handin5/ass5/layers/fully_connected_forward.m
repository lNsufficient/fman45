function Y = fully_connected_forward(X, A, b)
    % Inputs
    %   X - The input variable. The size might vary, but the last dimension
    %       tells you which element in the batch it is.
    %   A  - The weight matrix
    %   b  - The bias vector
    %
    % Outputs
    %   Y - The result as defined in the assignment instructions.
    
    % we take all dimension except the last one and vectorize them all so
    % that x has size [features x batchsize]
    sz = size(X);
    batch = sz(end);
    features = prod(sz(1:end-1));

    % suitable for matrix multiplication
    X = reshape(X, [features, batch]);

    assert(size(A, 2) == features, ...
        sprintf('Expected %d columns in the weights matrix, got %d', features, size(A,2)));
    assert(size(A, 1) == numel(b), 'Expected as many rows in A as elements in b');
    
    Y = [A, b]*[X; ones(1,size(X,2))];
    %error('Implement this');
end
