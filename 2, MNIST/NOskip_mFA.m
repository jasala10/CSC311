
function net = NOskip_mFA(net, delta_out)


n_m = size(delta_out, 2);   % number of samples in the current batch during training
l = net.n_layers;           % number of layers in the network
net.delta{l} = delta_out;   % error signal of the last layer, transpose of dL/dv

net.dL_dW{l} = net.delta{l}* (net.y{l-1})' /n_m;    
net.dL_db{l} = sum(net.delta{l}, 2)/n_m;

% define the derivatives wrt v, W, b respectively:

for l = net.n_layers-1:-1:2
    if l == 3 
        net.delta{l} = net.WB{l+1}' * net.delta{l+1} .* ...
            (max(0, sign(net.y{l})) .* (1./(1+net.y{l}))); 

    elseif l == 2
        % find layer two section that came from layer four:
        
        % find layer two section that came from layer three:

        delta_2_part2 = net.WB{l+1}' * net.delta{l+1} .* ...
            (max(0, sign(net.y{l})) .* (1./(1+net.y{l})));

        net.delta{l} = delta_2_part2;

    end
    net.dL_dW{l} = net.delta{l}*net.y{l - 1}'/n_m;
    net.dL_db{l} = sum(net.delta{l}, 2)/n_m;


    % PARTIAL ALIGNMENT
    eta(l) = 3/net.n_neurons(l - 1);
    p = 0.25;

    % Get the current gradients
    grad_W_FA = net.dL_dW{l};
    grad_b_FA = net.dL_db{l};

    % Flatten gradients and get the indices of the top p% largest values
    [~, idx_W] = sort(abs(grad_W_FA(:)), 'descend');
    [~, idx_b] = sort(abs(grad_b_FA(:)), 'descend');

    num_to_update_W = round(p * numel(grad_W_FA));
    num_to_update_b = round(p * numel(grad_b_FA));

    mask_W = false(size(grad_W_FA));
    mask_b = false(size(grad_b_FA));
    mask_W(idx_W(1:num_to_update_W)) = true;
    mask_b(idx_b(1:num_to_update_b)) = true;

    % Update only the selected weights
    net.W{l} = net.W{l} - eta(l) * (grad_W_FA .* mask_W);
    net.b{l} = net.b{l} - eta(l) * (grad_b_FA .* mask_b);

end

net.delta{1} = net.W{2}' * net.delta{2};

end





