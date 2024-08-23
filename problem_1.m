clear
clc
close all
%%
load('kuramoto_sivishinky.mat');  % Loads x, tt, uu
% Extract relevant variables
dt = tt(2) - tt(1);  % Time step from the data
% Define input-output data for training NN
input_data = uu(:, 1:end-1);  % Input: all columns except the last
output_data = uu(:, 2:end);   % Output: all columns except the first
% Neural Network Training (Task 1)
layers = [
    sequenceInputLayer(size(input_data, 1))
    lstmLayer(200, 'OutputMode', 'sequence')
    fullyConnectedLayer(size(output_data, 1))
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 50, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.8, ...
    'LearnRateDropPeriod', 10, ...
    'Verbose', true);
% Train the network
net = trainNetwork(input_data, output_data, layers, options);
% Simulate ODE Time-stepper and Compare with NN Predictions
N = 1024;
x = 32 * pi * (1:N)' / N;
u = cos(x/16) .* (1 + sin(x/16));
v = fft(u);
% Parameters for time-stepping
h = 0.025;
k = [0:N/2-1 0 -N/2+1:-1]' / 16;
L = k.^2 - k.^4;
E = exp(h * L); 
E2 = exp(h * L / 2);
M = 16;
r = exp(1i * pi * ((1:M) - 0.5) / M);
LR = h * L(:, ones(M, 1)) + r(ones(N, 1), :);
Q = h * real(mean((exp(LR / 2) - 1) ./ LR, 2));
f1 = h * real(mean((-4 - LR + exp(LR) .* (4 - 3 * LR + LR.^2)) ./ LR.^3, 2));
f2 = h * real(mean((2 + LR + exp(LR) .* (-2 + LR)) ./ LR.^3, 2));
f3 = h * real(mean((-4 - 3 * LR - LR.^2 + exp(LR) .* (4 - LR)) ./ LR.^3, 2));
% Main time-stepping loop to simulate ODE time-stepper
tmax = 100;
nmax = round(tmax / h);
nplt = floor((tmax / 250) / h);
g = -0.5i * k;
uu_ode = u;  % Store initial condition
tt_ode = 0;  % Time vector for ODE time-stepper
for n = 1:nmax
    t = n * h;
    Nv = g .* fft(real(ifft(v)).^2);
    a = E2 .* v + Q .* Nv;
    Na = g .* fft(real(ifft(a)).^2);
    b = E2 .* v + Q .* Na;
    Nb = g .* fft(real(ifft(b)).^2);
    c = E2 .* a + Q .* (2 * Nb - Nv);
    Nc = g .* fft(real(ifft(c)).^2);
    v = E .* v + Nv .* f1 + 2 * (Na + Nb) .* f2 + Nc .* f3;
    if mod(n, nplt) == 0
        u = real(ifft(v));
        uu_ode = [uu_ode, u];
        tt_ode = [tt_ode, t];
    end
end
%% Compare NN predictions and ODE time-stepper results
% Ensure time vectors are compatible for plotting
if length(tt) > length(tt_ode)
    tt = tt(1:length(tt_ode));
elseif length(tt) < length(tt_ode)
    tt_ode = tt_ode(1:length(tt));
end
%% figures
figure;
subplot(1, 2, 1);
plot(tt, uu(:, 1:length(tt)), 'LineWidth', 1.5);
hold on;
nn_predictions = predict(net, input_data);
plot(tt(2:end), nn_predictions(:, 1:length(tt)-1), '--', 'LineWidth', 1.5);
xlabel('Time');
ylabel('State');
title('Neural Network vs ODE Time-stepper');
legend('True Data', 'Neural Network Predictions');
grid on;
subplot(1, 2, 2);
plot(tt_ode, uu_ode(:, 1:length(tt_ode)), 'LineWidth', 1.5);
xlabel('Time');
ylabel('State');
title('ODE Time-stepper');
legend('True Data');
grid on;
% Forecasting in low-dimensional subspace via SVD
[U, S, V] = svd(uu);
rank_approx = 10;  % Example: using the first 10 singular values/components
uu_low_rank = U(:, 1:rank_approx) * S(1:rank_approx, 1:rank_approx) * V(:, 1:rank_approx)';
% Plotting low-rank approximation
figure;
subplot(1, 2, 1);
pcolor(x, tt, uu');
shading interp;
colormap(hot);
title('Original Data');
xlabel('Space');
ylabel('Time');
colorbar;
subplot(1, 2, 2);
pcolor(x, tt, uu_low_rank');
shading interp;
colormap(hot);
title(sprintf('Low-rank Approximation (Rank %d)', rank_approx));
xlabel('Space');
ylabel('Time');
colorbar;

