function net = training(net, x, labels, opts)
    loss = zeros(opts.iterations,1);
    loss_weight_decay = zeros(opts.iterations,1);
    loss_ma = zeros(opts.iterations,1);
    accuracy = zeros(opts.iterations,1);
    accuracy_ma = zeros(opts.iterations,1);

    sz = size(x);
    n_training = sz(end);
    % hack to access (:,:,???,:,elements in batch)
    x = reshape(x, [prod(sz(1:end-1)), n_training]);

    % might be useful
    momentum = cell(numel(net.layers),1);
    
    for it=1:opts.iterations
        tic;
        
        % extract the elements of the batch
        indices = 1+mod((it-1)*opts.batch_size:it*opts.batch_size-1, n_training);
        x_batch = reshape(x(:,indices), [sz(1:end-1) opts.batch_size]);
        labels_batch = labels(indices);

        % forward and backward pass
        [y, grads] = evaluate(net, x_batch, labels_batch);
        loss(it) = y{end};
        if isnan(loss(it)) || isinf(loss(it))
            error('Loss is NaN or inf. Decrease the learning rate or change the initialization.');
        end
        % we have a fully connected layer before the softmax loss
        % the prediction is the score that is highest
        [~,pred] = max(y{end-1}, [], 1);
        accuracy(it) = mean(vec(labels_batch) == vec(pred));
        if it < 20
            loss_ma(it) = mean(loss(1:it));
            accuracy_ma(it) = mean(accuracy(1:it));
        else
            loss_ma(it) = opts.moving_average*loss_ma(it-1) + ...
                (1 - opts.moving_average)*loss(it);
            accuracy_ma(it) = opts.moving_average*accuracy_ma(it-1) + ...
                (1 - opts.moving_average)*accuracy(it);
        end
        
        % gradient descent by looping over all parameters
        for i=2:numel(net.layers)
            layer = net.layers{i};
            
            % does the layer have any parameters? In that case we update
            if isfield(layer, 'params')
                params = fieldnames(layer.params);

                for k=1:numel(params)
                    s = params{k};
                    
                    % compute the weight decay loss
                    loss_weight_decay(it) = loss_weight_decay(it) + ...
                        opts.weight_decay/2*sum(vec(net.layers{i}.params.(s).^2));

                    % momentum and update
                    if isfield(opts, 'momentum')
                        % We loop over all layers and then all parameters.
                        % We use momentum{i}.(s) as the momentum for
                        % parameter s in layer number i. Note that theta in
                        % the assignment is just a convenient placeholder
                        % meaning all parameters in all layers. You can see
                        % the code for normal gradient descent below.
                        % Remember to include weight decay as param <- param
                        % - lr*(momentum + weight_decay*param)
                        %error('Implement the momentum update');
                        if it==1
                            momentum{i}.(s) = zeros(size(net.layers{i}.params.(s)));
                        end
                        mu = opts.momentum;
%                         if isempty(m_n)
%                             m_n = 
                        momentum{i}.(s) = mu*momentum{i}.(s) + (1-mu)*grads{i}.(s);
                        %m_n = mu*m_n+(1-mu)*grads{i};
                        net.layers{i}.params.(s) = net.layers{i}.params.(s) - ...
                            opts.learning_rate * (momentum{i}.(s) + ...
                                opts.weight_decay*net.layers{i}.params.(s));
                        %error('Implement this');
                    else
                        % run normal gradient descent if 
                        % the momentum parameter not is specified
                        net.layers{i}.params.(s) = net.layers{i}.params.(s) - ...
                            opts.learning_rate * (grads{i}.(s) + ...
                                opts.weight_decay * net.layers{i}.params.(s));
                    end
                end
            end
        end

        speed = opts.batch_size / toc;
        fprintf('It %d:\n', it);
        fprintf('Classification loss: %6f (%6f)\n', loss_ma(it), loss(it));
        fprintf('Weight decay loss: %6f\n', loss_weight_decay(it));
        fprintf('Total loss: %6f\n', loss_ma(it)+loss_weight_decay(it));
        fprintf('Accuracy %3f (%3f) %2.2f/s\n\n', ...
            accuracy_ma(it), accuracy(it), speed);
    end

    figure(1);
    plot(1:opts.iterations, loss_ma+loss_weight_decay);
    xlabel('Iteration');
    ylabel('Loss');

    figure(2);
    plot(1:opts.iterations, accuracy_ma);
    xlabel('Iteration');
    ylabel('Accuracy');
end
