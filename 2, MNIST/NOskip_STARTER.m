% Author: Jake Lance
% Date: November 29th

clear variables

% Load the MNIST data
load 'MNIST.mat';



n_iterations = 4;
n_epochs = 10;
incorrect_FA_all = zeros(n_iterations, n_epochs);
incorrect_mFA_all = zeros(n_iterations, n_epochs);
incorrect_BP_all = zeros(n_iterations, n_epochs);


for iter = 1:n_iterations
    disp(['Iteration ', num2str(iter)])

    % Create the network and define simulation variables
    y_FA = NOskip_mapnet();              % Feedback Alignment Net
    y_mFA = NOskip_mapnet();             % Modified FA Net
    y_BP = NOskip_mapnet();              % Backpropagation Net


    % ERROR CORRECTION (we want identical initialization):
    test_m = [0; 49; 49; 784]; % for convolution
    for l = 2:y_BP.n_layers % l = 2, 3, 4
        m = sqrt(2/test_m(l));
        y_mFA.W{l} = y_FA.W{l};
        y_BP.W{l} = y_FA.W{l};
    end    

    n_batches = 600;            % minibatches per epoch
    n_m = 100;                  % examples per minibatch 
    n_m_final = 10000;          % examples in the final testing minibatch
    eta = 0.001;                % learning rate
    
    % test_incorrect = zeros(10, 1);  % number incorrect per epoch of testing
    
    test_incorrect_BP = zeros(1, n_epochs);
    test_incorrect_FA = zeros(1, n_epochs);
    test_incorrect_mFA = zeros(1, n_epochs);


    for epoch = 1:n_epochs %epoch = 1:10
        
        % BEGIN TRAINING
    
        % start of epoch, so shuffle the data
        shuffle = randperm(size(TRAIN_images, 1));
        TRAIN_images = TRAIN_images(shuffle, :);
        TRAIN_answers = TRAIN_answers(shuffle, :);
        TRAIN_labels = TRAIN_labels(shuffle, :);
    
        incorrect_guesses_epoch = 0;
    
        disp(["a new epoch has begun"]) 
    
        for batch = 1:n_batches 
            % extract minibatch
            start_idx = (batch - 1) * n_m + 1;
            end_idx = start_idx + n_m - 1;
            minibatch_images = TRAIN_images(start_idx:end_idx, :);
            minibatch_labels = TRAIN_labels(start_idx:end_idx, :);
        
            % forward pass
            y_FA = NOskip_forward(y_FA, minibatch_images');
            y_mFA = NOskip_forward(y_mFA, minibatch_images');
            y_BP = NOskip_forward(y_BP, minibatch_images');
            
    
            output_FA = y_FA.y{4};
            output_mFA = y_mFA.y{4};
            output_BP = y_BP.y{4};
            
            soft_output_FA = exp(output_FA) ./ sum(exp(output_FA), 1);
            soft_output_mFA = exp(output_mFA) ./ sum(exp(output_mFA), 1);
            soft_output_BP = exp(output_BP) ./ sum(exp(output_BP), 1);

     

            % DOUBLE CHECK: calculate error and check for incorrect guesses
            [~, predicted_labels_FA] = max(soft_output_FA);
            [~, true_labels_FA] = max(minibatch_labels');
            incorrect_FA(epoch) = sum(predicted_labels_FA ~= true_labels_FA);
    
            [~, predicted_labels_mFA] = max(soft_output_mFA);
            [~, true_labels_mFA] = max(minibatch_labels');
            incorrect_mFA(epoch) = sum(predicted_labels_mFA ~= true_labels_mFA);
    
    
            [~, predicted_labels_BP] = max(soft_output_BP);
            [~, true_labels_BP] = max(minibatch_labels');
            incorrect_BP(epoch) = sum(predicted_labels_BP ~= true_labels_BP);
    
    
            % calculate error and loss
            e_FA = soft_output_FA - TRAIN_labels(start_idx:end_idx, :)'; 
            e_mFA = soft_output_mFA - TRAIN_labels(start_idx:end_idx, :)';         
            e_BP = soft_output_BP - TRAIN_labels(start_idx:end_idx, :)';  
            
            L_FA = 0.5*(e_FA*e_FA');
            L_mFA = 0.5*(e_mFA*e_mFA');
            L_BP = 0.5*(e_BP*e_BP');
    
    
            % backward pass
            % y = pg_backprop_relog(y, e);      
            y_FA = NOskip_FA(y_FA, e_FA); 
            y_mFA = NOskip_mFA(y_mFA, e_mFA);
            y_BP = NOskip_backprop(y_BP, e_BP); 
    
            % adam
            y_FA = adam(y_FA, y_FA.dL_dW, y_FA.dL_db, eta, 0.9, 0.999);
            y_mFA = adam(y_mFA, y_mFA.dL_dW, y_mFA.dL_db, eta, 0.9, 0.999);
            y_BP = adam(y_BP, y_BP.dL_dW, y_BP.dL_db, eta, 0.9, 0.999);
        
        end % TRAINING
        
        % BEGIN TESTING
    
        % we run the network on the entire test set, in one "minibatch" 
        % of 10,000 test examples 
    
        shuffle = randperm(size(TEST_images, 1));
        TEST_images = TEST_images(shuffle, :);
        TEST_answers = TEST_answers(shuffle, :);
        TEST_labels = TEST_labels(shuffle, :);
        
        start_idx = 1;
        end_idx = 10000;
        minibatch_images = TEST_images(start_idx:end_idx, :);
        minibatch_answers = TEST_answers(start_idx:end_idx, :);
        minibatch_labels = TEST_labels(start_idx:end_idx, :);
    
        % forward pass
        y_FA = NOskip_forward(y_FA, minibatch_images');
        y_mFA = NOskip_forward(y_mFA, minibatch_images');
        y_BP = NOskip_forward(y_BP, minibatch_images');
    
        output_FA = y_FA.y{4};
        soft_output_FA = exp(output_FA) ./ sum(exp(output_FA), 1);
    
        output_mFA = y_mFA.y{4};
        soft_output_mFA = exp(output_mFA) ./ sum(exp(output_mFA), 1);
    
        output_BP = y_BP.y{4};
        soft_output_BP = exp(output_BP) ./ sum(exp(output_BP), 1);
    
    
        % check for incorrect guesses
        [~, predicted_labels_FA] = max(soft_output_FA);
        [~, true_labels_FA] = max(minibatch_labels');
        test_incorrect_FA(epoch) = sum(predicted_labels_FA ~= true_labels_FA);
    
        [~, predicted_labels_mFA] = max(soft_output_mFA);
        [~, true_labels_mFA] = max(minibatch_labels');
        test_incorrect_mFA(epoch) = sum(predicted_labels_mFA ~= true_labels_mFA);
    
        [~, predicted_labels_BP] = max(soft_output_BP);
        [~, true_labels_BP] = max(minibatch_labels');
        test_incorrect_BP(epoch) = sum(predicted_labels_BP ~= true_labels_BP);

        
            
    end % epochs

    incorrect_FA_all(iter, :) = test_incorrect_FA;
    incorrect_mFA_all(iter, :) = test_incorrect_mFA;
    incorrect_BP_all(iter, :) = test_incorrect_BP;


end % iterations


% Average results across all runs
avg_test_incorrect_FA = mean(incorrect_FA_all, 1);
avg_test_incorrect_mFA = mean(incorrect_mFA_all, 1);
avg_test_incorrect_BP = mean(incorrect_BP_all, 1);



% disp([incorrect; "incorrect"])
disp("test_incorrect_FA:")
disp([test_incorrect_FA]') 
disp("test_incorrect_mFA:")
disp([test_incorrect_mFA]') 
disp("test_incorrect_BP:")
disp([test_incorrect_BP]') 


% Plot results
figure;
hold on;
plot(1:n_epochs, avg_test_incorrect_BP, 'b-o', 'LineWidth', 2, 'DisplayName', 'BP');
plot(1:n_epochs, avg_test_incorrect_FA, 'r-o', 'LineWidth', 2, 'DisplayName', 'FA');
plot(1:n_epochs, avg_test_incorrect_mFA, 'Color', [1, 0.6, 0.8], 'LineWidth', 2, 'DisplayName', 'mFA'); % pink
xlabel('Epoch');
ylabel('Average Number of Incorrect Predictions');
title('Comparison of BP, FA, and mFA on Test Data');
legend('Location', 'best');
grid on;
hold off;


%{

test_incorrect_FA:
   639
   411
   330
   283
   232
   220
   198
   191
   203
   193

test_incorrect_mFA:
   635
   400
   306
   265
   246
   239
   206
   196
   205
   188

test_incorrect_BP:
   418
   265
   229
   194
   186
   167
   173
   166
   180
   158

%}
