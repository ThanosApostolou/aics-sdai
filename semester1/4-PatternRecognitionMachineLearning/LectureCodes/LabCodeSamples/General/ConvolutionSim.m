% Define the number of samples for both the signal and the impulse
% response.
N = 128;
% Define the interval corresponding to the input signal and the impulse
% response.
n = [0:1:N-1];
% Define the signal and the impulse response.
x = sin(pi+(pi/64)*n);
h = exp(0.01*n);
% Plot the signal.
figure('Name','Input Signal')
stem(n,x,'-r','LineWidth',1.5);
xlabel('Time Interval');
ylabel('Signal Value')
grid on
% Plot the impulse response.
figure('Name','Impulse Response')
stem(n,h,'-r','LineWidth',1.8);
xlabel('Time Interval');
ylabel('Signal Value')
grid on
% Compute the output signal.
y = myconv(x,h)
% Plot input signal, impulse response and ouput signal in the same window.
figure('Name','Output Signal')
subplot(3,1,1)
stem(n,x,'-r','LineWidth',1.5);
xlabel('Time Interval');
ylabel('Signal Value')
grid on
subplot(3,1,2)
stem(n,h,'-b','LineWidth',1.8);
xlabel('Time Interval');
ylabel('Signal Value')
grid on
subplot(3,1,3)
stem(n,y,'-g','LineWidth',1.8);
xlabel('Time Interval');
ylabel('Signal Value')
grid on