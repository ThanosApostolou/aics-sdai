% Approximating the logical function corresponding to the OR gate by
% utilizing  a single layer Perceptron.
%-------------------------------------------------------------------------
%-------------------------------------------------------------------------
% Solving the problem by incrementally training the Perceptron.
%-------------------------------------------------------------------------
% When using incremental training all training vectors and and patterns
% must be stored in a cell array.
% Setting training vectors and corresponding targets within the cell arrays
% P and T for incremental learning.
clear all
clc
P = {[0;0],[0;1],[1;0],[1;1]}
T = {0,1,1,1}
% Create the perceptron and store it inside a net object.
net = newp([0 1;0 1],1)
% Simulate perceptron response before incremental learning.
Y = sim(net,P)
% Let the network adapt for 10 passes through the sequence.
net.adaptParam.passes = 10
net = adapt(net,P,T)
% Simulate perceptron response after incremental learning.
Y = sim(net,P)
% Perceptron internal parameters.
W = net.IW{1,1}
b = net.b{1}
%-------------------------------------------------------------------------
% Solving the same problem by utilizing batch training.
% Batch training requires that both training vectors and corresponfing 
% targets are stored in concurrent vectors.
% Setting training vectors and corresponding targets within the concurrent
% matrix P and vector T for batch training.
P = [0 0 1 1;0 1 0 1];
T = [0 1 1 1];
% Initialize network object.
net = init(net)
% Simulate perceptron response before batch training.
Y = sim(net,P)
% Train perceptron for a maximum of 20 epochs.
net.trainParam.epochs = 20;
% Train network in batch mode.
net = train(net,P,T)
% Simulate perceptron response after batch training.
Y = sim(net,P)
% Perceptron internal parameters.
W = net.IW{1,1}
b = net.b{1}