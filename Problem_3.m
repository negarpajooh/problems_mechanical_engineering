clear all;    
close all;    
clc;         
%% data
% Parameters
N = 256;            % Reduced dimension size (was 512)
T = 100;            % Reduced time steps (was 201)
% Generate data
t = linspace(0, 10, T);      % Time vector
x = linspace(0, 1, N);       % Space vector x
y = linspace(0, 1, N);       % Space vector y
% Create random data for u and v
u = randn(N, N, T);          % u: 256x256x100
v = randn(N, N, T);          % v: 256x256x100
% Reshape u and v for NN input
u_flat = reshape(u, [], T)';    % Reshape to T x (N*N)
v_flat = reshape(v, [], T)';    % Reshape to T x (N*N)
%% process
% Perform SVD on u and v
[Uu, Su, Vu] = svd(u_flat, 'econ');
[Uv, Sv, Vv] = svd(v_flat, 'econ');
% Train RNN (example using MATLAB's built-in functions)
inputSize = N*N;        % Input size
numHiddenUnits = 100;   % Number of hidden units
numClasses = N*N;       % Output size
% Define network architecture
layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(numClasses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.005, ...
    'MiniBatchSize', 32, ... % Adjust batch size if needed
    'Verbose', 0, ...
    'Plots', 'training-progress');
% Train network using batch processing
net = trainNetwork(u_flat', v_flat', layers, options);
% Visualization (example plot)
figure;
subplot(1, 2, 1);
imagesc(u(:, :, 1));   % Example plot of u at time step 1
title('Matrix u at Time Step 1');
colorbar;
subplot(1, 2, 2);
imagesc(v(:, :, 1));   % Example plot of v at time step 1
title('Matrix v at Time Step 1');
colorbar; 
%% Generate predictions using trained networks or methods
% Example: Predict using trained neural network (net)
v_pred_nn = predict(net, u_flat')';
% Reshape predictions back to 3D matrices
v_pred_nn_3d = reshape(v_pred_nn, [N, N, T]);
% Example: Plot comparison of original v and predicted v using neural network
figure;
subplot(2, 2, 1);
imagesc(v(:, :, 1));   % Original v at time step 1
title('Original v at Time Step 1');
colorbar;
subplot(2, 2, 2);
imagesc(v_pred_nn_3d(:, :, 1));   % Predicted v at time step 1 using NN
title('Predicted v using NN at Time Step 1');
colorbar;
% Plot SVD components and reconstructions
subplot(2, 2, 3);
plot(diag(Sv), 'o');    % Singular values of v
title('Singular Values of v');
xlabel('Singular Value Index');
ylabel('Singular Value');
% Reconstruct v using reduced components
k = 10;  % Number of singular values to keep
v_reconstructed = Uv(:, 1:k) * Sv(1:k, 1:k) * Vv(:, 1:k)';
v_reconstructed_3d = reshape(v_reconstructed', [N, N, T]);
%%
subplot(2, 2, 4);
imagesc(v_reconstructed_3d(:, :, 1));   % Reconstructed v at time step 1
title(sprintf('Reconstructed v (k=%d) at Time Step 1', k));
colorbar;
