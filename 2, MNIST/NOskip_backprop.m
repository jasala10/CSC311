
function net = NOskip_backprop(net, delta_out)


n_m = size(delta_out, 2);   % number of samples in the current batch during training
l = net.n_layers;           % number of layers in the network
net.delta{l} = delta_out;   % error signal of the last layer, transpose of dL/dv

net.dL_dW{l} = net.delta{l}* (net.y{l-1})' /n_m;    
net.dL_db{l} = sum(net.delta{l}, 2)/n_m;

% define the derivatives wrt v, W, b respectively:

for l = net.n_layers-1:-1:2
    if l == 3 
        net.delta{l} = net.W{l+1}' * net.delta{l+1} .* ...
            (max(0, sign(net.y{l})) .* (1./(1+net.y{l}))); 

    elseif l == 2
        % find layer two section that came from layer four:
        
        % find layer two section that came from layer three:

        delta_2_part2 = net.W{l+1}' * net.delta{l+1} .* ...
            (max(0, sign(net.y{l})) .* (1./(1+net.y{l})));

        net.delta{l} = delta_2_part2;

    end
    net.dL_dW{l} = net.delta{l}*net.y{l - 1}'/n_m;
    net.dL_db{l} = sum(net.delta{l}, 2)/n_m;
end

net.delta{1} = net.W{2}' * net.delta{2};

end





