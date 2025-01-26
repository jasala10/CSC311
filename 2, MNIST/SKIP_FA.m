function net = SKIP_FA(net, delta_out)

% IDEA:
% start with the original backprop function you call to, adjust it to
% principles of FA


n_m = size(delta_out, 2);   % number of samples in the current batch during training
l = net.n_layers;           % number of layers in the network
net.delta{l} = delta_out;   % (v) error signal of the last layer, transpose of dL/dv
eta = 0.001;

net.dL_dW{l} = net.delta{l}* ([net.y{l - 2};net.y{l-1}])' /n_m;     % EDIT: just want from last layer
net.dL_db{l} = sum(net.delta{l}, 2)/n_m;                            

% define the derivatives wrt v, W, b respectively:

% UPDATES : anything with W changes to WB

for l = net.n_layers-1:-1:2
    if l == 3 
        net.delta{l} = (net.WB{l+1}(:, 785:end))' * net.delta{l+1} .* ...
            (max(0, sign(net.y{l})) .* (1./(1+net.y{l}))); 

    elseif l == 2
        % find layer two section that came from layer four:
        
        top_of_WB4 = net.WB{4}(:, 1:784)'; % bottom part of W{4}' direct to 2
        delta_2_part1 = top_of_WB4 * net.delta{4} .* ...
            (max(0, sign(net.y{l})) .* (1./(1+net.y{l})));
        
        % find layer two section that came from layer three:

        delta_2_part2 = net.WB{l+1}' * net.delta{l+1} .* ...
            (max(0, sign(net.y{l})) .* (1./(1+net.y{l})));

        net.delta{l} = delta_2_part1 + delta_2_part2;

    end
    net.dL_dW{l} = net.delta{l}*net.y{l - 1}'/n_m;
    net.dL_db{l} = sum(net.delta{l}, 2)/n_m;
end

net.delta{1} = net.W{2}' * net.delta{2};

% ADJUST PARAMETERS
for l = 2:net.n_layers
    net.W{l} = net.W{l} - eta*net.dL_dW{l};
    net.b{l} = net.b{l} - eta*net.dL_db{l};
end  

end





