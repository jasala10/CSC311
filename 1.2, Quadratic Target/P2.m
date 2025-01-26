%{

Date: November 29th, 2024
Author: Jake Lance

Phase 1, Step 2:
BP vs mFA with Quadratic Learning Task


%}


clear variables;
clc;
clf;

% Initialize
Dt = 1;                         % time step
dur = 1e4;                      % simulation duration
n_neurons = [5; 30; 2];         % neurons in layers
n_layers = numel(n_neurons);    % number of layers
p = 0.5;



maxi = 300;
error_vec = zeros(maxi, 1);

for epoch = 1:maxi

  

% ~~~~ ERROR STATISTICS ~~~ %



y_star_sum_FA = zeros(n_neurons(end), 1);         
y_star_sq_sum_FA = zeros(n_neurons(end), 1);       

y_star_sum_BP = zeros(n_neurons(end), 1);          
y_star_sq_sum_BP = zeros(n_neurons(end), 1);     


% ~~~ WEIGHTS AND BIASES ~~~ %


% ORIGINAL Weights/Biases FOR FA, renamed
[W_FA, dL_dW_FA, b_FA, dL_db_FA, y_FA, ...
    target_FA, dL_dv_FA, WB_FA] = deal(cell(n_layers, 1));

% ADD STUFF FOR BP
[W_BP, dL_dW_BP, b_BP, dL_db_BP, y_BP, ...
    dL_dv_BP] = deal(cell(n_layers, 1));

B = rand(n_neurons(n_layers), n_neurons(1)) - 0.5; 



% ~~~ LEARNING RATE ~~~ %
eta = zeros(n_layers, 1);                      



% ~~~ SET UP LAYERS ~~~ %


 
    % FA
    for l = 2:n_layers
      m = 0.1/sqrt(n_neurons(l - 1));                           % scaling down for weight initialization
      W_FA{l} = m*(rand(n_neurons(l), n_neurons(l - 1)) - 0.5);    % forward weight initialization
      dL_dW_FA{l} = zeros(n_neurons(l), n_neurons(l - 1));         % gradient of weights 
      b_FA{l} = 0.1*rand(n_neurons(l), 1);                         % forward bias initialization
      dL_db_FA{l} = zeros(n_neurons(l), 1);                        % gradient of biases
      WB_FA{l} = m*(rand(n_neurons(l), n_neurons(l - 1)) - 0.5);   % substitute weight matrix
      eta(l) = 3/n_neurons(l - 1);                                 % learning rate
    end
    
    % BP
    for l = 2:n_layers
      m = 0.1/sqrt(n_neurons(l - 1));                           % scaling down for weight initialization
      W_BP{l} = m*(rand(n_neurons(l), n_neurons(l - 1)) - 0.5);    % forward weight initialization
      dL_dW_BP{l} = zeros(n_neurons(l), n_neurons(l - 1));         % gradient of weights 
      b_BP{l} = 0.1*rand(n_neurons(l), 1);                         % forward bias initialization
      dL_db_BP{l} = zeros(n_neurons(l), 1);                        % gradient of biases
      % NO WB MATRIX
      eta(l) = 3/n_neurons(l - 1);                              % learning rate
    end
    
    
    % ~~~ TRAINING PARAMETERS ~~~ %
    
    n_steps = 1 + floor(dur/Dt);
    gt_step = ceil(n_steps/1000);  % graphical step, ensures <= 1000 pts per plot
    DATA = zeros(1 + 2*n_neurons(end), 1 + floor(n_steps/gt_step));  % allocate more than enough memory
    step = 0;
    i_gp = 0;  % index of graph points
    
    % Train
    
    for t = 0:Dt:dur
        
        step = step + 1;  
        
        % Set input and desired output
        x = sqrt(3)*(rand(n_neurons(1), 1) - 0.5);        % choose x with SD 0.5
        y_star = (B*x + 1).^2;                            % quadratic TARGET FUNCTION
          
        y_star_sum_FA = y_star_sum_FA + y_star;                 % statistics of y_star
        y_star_sq_sum_FA = y_star_sq_sum_FA + y_star.*y_star;   % """ """ ""
        y_star_sum_BP = y_star_sum_BP + y_star;
        y_star_sq_sum_BP = y_star_sq_sum_BP + y_star.*y_star;
        
         
        % FA
        y_FA{1} = x;                                                         % define input layer
        for l = 2:n_layers - 1
            y_FA{l} = max(0, W_FA{l}*y_FA{l - 1} + b_FA{l});                            % relu activation
        end
        y_FA{n_layers} = W_FA{n_layers}*y_FA{n_layers - 1} + b_FA{n_layers};          % neurons in final layer are affine
        e_FA = y_FA{n_layers} - y_star;  % error
        

 

        % BP
        y_BP{1} = x;
        for l = 2:n_layers - 1
          y_BP{l} = max(0, W_BP{l}*y_BP{l - 1} + b_BP{l});
        end
        y_BP{n_layers} = W_BP{n_layers}*y_BP{n_layers - 1} + b_BP{n_layers};
        e_BP = y_BP{n_layers} - y_star;
    

        % FA
        l = n_layers;
        dL_dv_FA{l} = e_FA;         % actually the transpose of dL/dv. You can derive this from v{last} = y{last} and L=0.5e^2
        dL_dW_FA{l} = dL_dv_FA{l} * y_FA{l - 1}';      % Chain rule for 0.5(W(n)y(n-1)' + b(n) - y*)^2
        dL_db_FA{l} = dL_dv_FA{l};                  
        for l = n_layers - 1:-1:2
            % dL_dv{l} = (WB{l + 1}'*dL_dv{l + 1}) .* sign(y{l});   % use WB instead of W
            dL_dv_FA{l} = (WB_FA{l + 1}'*dL_dv_FA{l + 1}) .* sign(y_FA{l});        
            %dL_dv_FA{l} = dL_dv_FA{l} + 2 * lambda * y_FA{l}; % REGULARIZATION
            
            dL_dW_FA{l} = dL_dv_FA{l} * y_FA{l - 1}';
            dL_db_FA{l} = dL_dv_FA{l};
        end
    
      % BP
        l = n_layers;
        dL_dv_BP{l} = e_BP;         % actually the transpose of dL/dv. You can derive this from v{last} = y{last} and L=0.5e^2
        dL_dW_BP{l} = dL_dv_BP{l} * y_BP{l - 1}';      % Chain rule for 0.5(W(n)y(n-1)' + b(n) - y*)^2
        dL_db_BP{l} = dL_dv_BP{l};                  
        for l = n_layers - 1:-1:2
            % dL_dv{l} = (WB{l + 1}'*dL_dv{l + 1}) .* sign(y{l});   % use WB instead of W
            dL_dv_BP{l} = (W_BP{l + 1}'*dL_dv_BP{l + 1}) .* sign(y_BP{l});        
            dL_dW_BP{l} = dL_dv_BP{l} * y_BP{l - 1}';
            dL_db_BP{l} = dL_dv_BP{l};
        end
    
    
        % ~~~ ADJUST PARAMETERS ~~~
    
        
        % ADAPTED, RENAMED FOR FA
        for l = 2:n_layers
            
            % % Get the current gradients
            grad_W_FA = dL_dW_FA{l};
            grad_b_FA = dL_db_FA{l};

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
            W_FA{l} = W_FA{l} - eta(l) * (grad_W_FA .* mask_W);
            b_FA{l} = b_FA{l} - eta(l) * (grad_b_FA .* mask_b);


            % OLD UPDATE
            % W_FA{l} = W_FA{l} - eta(l)*dL_dW_FA{l};
            % b_FA{l} = b_FA{l} - eta(l)*dL_db_FA{l};
        end
        
        % ADAPTED, BP
        for l = 2:n_layers
            W_BP{l} = W_BP{l} - eta(l)*dL_dW_BP{l};
            b_BP{l} = b_BP{l} - eta(l)*dL_db_BP{l};
        end
    
    
      % Record data for plotting
        if mod(step, gt_step) == 0
            i_gp = i_gp + 1;
            DATA(:, i_gp) = [t; e_FA.*e_FA; e_BP.*e_BP];
           
        end
    
    end  % for t
    DATA = DATA(:, 1:i_gp);
    
    % Compute statistics of y
    y_star_avg_FA = y_star_sum_FA/step;
    y_star_var_FA = y_star_sq_sum_FA/step - y_star_avg_FA.*y_star_avg_FA;  % approximate
    DATA(2:n_neurons(end)+1, :) = DATA(2:n_neurons(end)+1, :) ./ y_star_var_FA;  % normalized squared error, NSE
    
    y_star_avg_BP = y_star_sum_BP/step;
    y_star_var_BP = y_star_sq_sum_BP/step - y_star_avg_BP.*y_star_avg_BP;
    DATA(n_neurons(end)+2:end, :) = DATA(n_neurons(end)+2:end, :) ./ y_star_var_BP; 
    
    
    % Plot
    figure(1);
    set(gcf, 'Name', 'Feedback alignment vs Backpropagation', 'NumberTitle', 'off');
    
    % % Plot 1: NSE for Feedback Alignment (linear scale)
    % subplot(4, 1, 1);
    % plot(DATA(1, :), DATA(2, :), 'r');  % NSE for Feedback Alignment
    % ylabel('\it{NSE (FA)}');
    % line([DATA(1, 1), DATA(1, end)], [0.01, 0.01], 'LineStyle', '--', 'Color', 'k');
    % ylim([0, 0.1]);
    % xlim([DATA(1, 1), DATA(1, end)]);
    % set(gca, 'TickLength', [0, 0]);
    % 
    % % Plot 2: NSE for Backpropagation (linear scale)
    % subplot(4, 1, 2);
    % plot(DATA(1, :), DATA(4, :), 'b');  % NSE for Backpropagation
    % ylabel('\it{NSE (BP)}');
    % line([DATA(1, 1), DATA(1, end)], [0.01, 0.01], 'LineStyle', '--', 'Color', 'k');
    % ylim([0, 0.1]);
    % xlim([DATA(1, 1), DATA(1, end)]);
    % set(gca, 'TickLength', [0, 0]);
    % 
    % Plot 3: NSE for Feedback Alignment (log scale)
    % title('Backpropagation vs. Modified Feedback Alignment: Preliminary Task')
    % subplot(2, 1, 1);
    % loglog(DATA(1, :), DATA(2, :), 'r');  % NSE for Feedback Alignment (log scale)
    % ylabel('\it{NSE (mFA)}');
    % xlabel('\it{time steps}');
    % line([DATA(1, 1), DATA(1, end)], [0.01, 0.01], 'LineStyle', '--', 'Color', 'k');
    % ylim([1e-10, 1e2]);
    % xlim([DATA(1, 1), DATA(1, end)]);
    % set(gca, 'TickLength', [0, 0]);
    % 
    % % Plot 4: NSE for Backpropagation (log scale)
    % subplot(2, 1, 2);
    % loglog(DATA(1, :), DATA(4, :), 'b');  % NSE for Backpropagation (log scale)
    % ylabel('\it{NSE (BP)}');
    % xlabel('\it{time steps}');
    % line([DATA(1, 1), DATA(1, end)], [0.01, 0.01], 'LineStyle', '--', 'Color', 'k');
    % ylim([1e-10, 1e2]);
    % xlim([DATA(1, 1), DATA(1, end)]);
    % set(gca, 'TickLength', [0, 0]);
    % % 
    
    FA_error = sum(DATA(2, :));
    BP_error = sum(DATA(n_neurons(end)+2, :));
    error_diff = FA_error - BP_error;
    % disp('FA error');
    % disp(FA_error);
    % disp('BP error');
    % disp(BP_error);
    % disp('Difference:')
    % disp(FA_error - BP_error)
    
    error_vec(epoch) = error_diff; 

end % EPOCH

mean_error = mean(error_vec);
std_error = std(error_vec);


% 

% disp(error_vec)
% disp(sum(error_vec))
% Display the results
disp(['Mean Error Difference (FA - BP): ', num2str(mean_error)]);
disp(['Standard Deviation of Error Differences: ', num2str(std_error)]);
% Create the histogram
figure;
histogram(error_vec, 30, 'FaceColor', [0, 0.5, 0]); % 30 bins for the histogram (adjust as needed)
xlabel('Error Difference (mFA - BP)');
ylabel('Frequency');
title('Distribution of Error Differences');

% Optionally, display the mean and SD on the histogram plot
text(mean_error, max(ylim)*0.9, ['Mean: ', num2str(mean_error)], 'HorizontalAlignment', 'left', 'FontSize', 12, 'Color', 'r');
text(mean_error + std_error, max(ylim)*0.8, ['SD: ', num2str(std_error)], 'HorizontalAlignment', 'left', 'FontSize', 12, 'Color', 'r');



% 
% % % Plot
% figure(1);
% set(gcf, 'Name', 'Feedback alignment', 'NumberTitle', 'off');
% subplot(2, 1, 1);
% %plot(DATA(1, :), DATA(2, :), 'r');
% plot(DATA(1, :), max(DATA(2:3, :)), 'r');
% ylabel('\it{NSE}');
% line([DATA(1, 1), DATA(1, end)], [0.01, 0.01], 'LineStyle', '--', 'Color', 'k');
% ylim([0, 0.1]);
% xlim([DATA(1, 1), DATA(1, end)]);
% set(gca, 'TickLength', [0, 0]);
% 
% subplot(2, 1, 2);
% %loglog(DATA(1, :), DATA(2, :), 'r');
% loglog(DATA(1, :), max(DATA(2:3, :)), 'r');
% ylabel('\it{NSE}');
% xlabel('\it{t}');
% line([DATA(1, 1), DATA(1, end)], [0.01, 0.01], 'LineStyle', '--', 'Color', 'k');
% ylim([1e-10, 1e2]);
% xlim([DATA(1, 1), DATA(1, end)]);
% set(gca, 'TickLength', [0, 0]);
