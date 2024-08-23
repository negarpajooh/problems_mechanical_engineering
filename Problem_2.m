clear all;
close all;
clc;
%% equation
% Kuramoto-Sivashinsky equation (from Trefethen)
% u_t = -u*u_x - u_xx - u_xxxx 
% Parameters and initial conditions
N = 1024; % Number of grid points
x = 32*pi*(1:N)'/N; % Spatial grid
u_original = cos(x/16).*(1+sin(x/16)); % Original initial condition
v_original = fft(u_original); % Fourier transform of original initial condition
% Spatial grid and numerical parameters
h = 0.025; % Time step
k = [0:N/2-1 0 -N/2+1:-1]'/16; % Wavenumber vector
L = k.^2 - k.^4; % Fourier multipliers for spatial derivatives
E = exp(h*L); E2 = exp(h*L/2); % Time-stepping operators
M = 16; % Number of points for exponential integrator
r = exp(1i*pi*((1:M)-.5)/M); % Exponential integrator points
LR = h*L(:,ones(M,1)) + r(ones(N,1),:); % Matrix for exponential integrator
Q = h*real(mean( (exp(LR/2)-1)./LR ,2)); % Quadratic nonlinear term
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)); % Cubic nonlinear term
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2)); % Cubic nonlinear term
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2)); % Cubic nonlinear term
%% Main time-stepping loop for original initial condition
uu_original = u_original; % Array to store solution for original condition
tt = 0; % Time array initialization
tmax = 100; % Maximum time
nmax = round(tmax/h); % Number of time steps
nplt = floor((tmax/250)/h); % Plotting interval
g = -0.5i*k; % Nonlinear term in Fourier space

for n = 1:nmax
    t = n*h; % Current time
    Nv = g.*fft(real(ifft(v_original)).^2); % Nonlinear term in Fourier space
    a = E2.*v_original + Q.*Nv; % Exponential integrator step
    Na = g.*fft(real(ifft(a)).^2); % Nonlinear term in Fourier space
    b = E2.*v_original + Q.*Na; % Exponential integrator step
    Nb = g.*fft(real(ifft(b)).^2); % Nonlinear term in Fourier space
    c = E2.*a + Q.*(2*Nb-Nv); % Exponential integrator step
    Nc = g.*fft(real(ifft(c)).^2); % Nonlinear term in Fourier space
    v_original = E.*v_original + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; % Time-stepping update
    
    % Store solution for plotting
    if mod(n,nplt)==0
        u_original = real(ifft(v_original)); % Inverse Fourier transform to get u
        uu_original = [uu_original,u_original]; % Store solution
        tt = [tt,t]; % Update time array
    end
end
%% Comparing original and modified initial conditions
div1 = 20*rand() + 5; % Random divisor 1
div2 = 20*rand() + 5; % Random divisor 2
u_modified = cos(x/div1).*(1+sin(x/div2)); % Modified initial condition
v_modified = fft(u_modified); % Fourier transform of modified initial condition
save('kuramoto_sivishinky_original.mat','x','tt','uu_original')
u_modified = u_modified; % Set initial condition to modified
v_modified = fft(u_modified); % Fourier transform of modified initial condition
uu_modified = u_modified; tt = 0; % Reset storage arrays
for n = 1:nmax
    t = n*h; % Current time
    Nv = g.*fft(real(ifft(v_modified)).^2); % Nonlinear term in Fourier space
    a = E2.*v_modified + Q.*Nv; % Exponential integrator step
    Na = g.*fft(real(ifft(a)).^2); % Nonlinear term in Fourier space
    b = E2.*v_modified + Q.*Na; % Exponential integrator step
    Nb = g.*fft(real(ifft(b)).^2); % Nonlinear term in Fourier space
    c = E2.*a + Q.*(2*Nb-Nv); % Exponential integrator step
    Nc = g.*fft(real(ifft(c)).^2); % Nonlinear term in Fourier space
    v_modified = E.*v_modified + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; % Time-stepping update
    % Store solution for plotting
    if mod(n,nplt)==0
        u_modified = real(ifft(v_modified)); % Inverse Fourier transform to get u
        uu_modified = [uu_modified,u_modified]; % Store solution
        tt = [tt,t]; % Update time array
    end
end
%% figures
figure;
subplot(3,3,[1,2,3]);
surf(tt,x,uu_original), shading interp, colormap(hot), axis tight;
title('Original Initial Condition');
xlabel('Time');
ylabel('Space');
zlabel('u');
set(gca,'zlim',[-5 50]);
%
subplot(3,3,[4,5,6]);
surf(tt,x,uu_modified), shading interp, colormap(hot), axis tight;
title('Modified Initial Condition');
xlabel('Time');
ylabel('Space');
zlabel('u');
set(gca,'zlim',[-5 50]);
%
slice_times = [10, 50, 90]; % Time slices to plot
for i = 1:length(slice_times)
    [~, idx] = min(abs(tt - slice_times(i))); % Find closest time index 
    % Adjusting subplot indices to ensure they are within the range 1 to 9
    subplot(3,3,i+6); % For original condition
    plot(x, uu_original(:,idx));   
hold on
    subplot(3,3,i+6); % For modified condition
    plot(x, uu_modified(:,idx));
    title(['Time = ', num2str(slice_times(i))]);
    legend('Original','Modified')
    xlabel('Space');
    ylabel('u');
    grid on;
end
%
figure;
pcolor(x, tt, uu_modified.'), shading interp, colormap(hot), axis off;
title('Modified Initial Condition - pcolor');
xlabel('Space');
ylabel('Time');
