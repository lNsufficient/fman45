function dldx = softmaxloss_backward(x, labels)
    % Inputs:
    %    x - Features. See the reshape command below. It is reshaped as for
    %        the fully connected layer.
    %    y - Labels. It is a vector with the correct labels. For
    %        instance if we have a batch of two where the first example is
    %        class 4 and the second example is class 7, labels is [4 7].
    %
    % Outputs:
    %    dldx - Partial derivative of L with respect to x. Remember that in
    %           the forward pass you average over the batch elements.
    sz = size(x);
    batch = sz(end);
    features = prod(sz(1:end-1));

    % suitable for matrix multiplication
    x = reshape(x, [features, batch]);
    % for numerical stability. Convince yourself that the result is the same.
    x = bsxfun(@minus, x, min(x, [], 1));

    %error('Implement this function');
    labels = labels(:);
    IND = labels' + uint8(sz(1)*(0:batch-1));
    row_factors = 1./(sum(exp(x),1));
    dldx = 1/batch*exp(x)*(diag(row_factors));
    dldx(IND) = dldx(IND) - 1/batch;
end
