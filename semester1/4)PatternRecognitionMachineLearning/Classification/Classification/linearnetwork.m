% Denonstration of training styles of Linear Neural Networks.
% Suppose that we want to train a network to create the linear function t =
% t = 2 * p1 + p2.
%-------------------------------------------------------------------------
% Firstly, we set up the network with zero initial weights and biases. We
% also set the learning rate to zero initially to show the effect of
% incremental training.
%-------------------------------------------------------------------------
%-------------------------------------------------------------------------
clc
clear all
net  = newlin([-1 1;-1 1],1,0,0)
net.IW{1,1} = [0 0]
net.b{1} = 0
%-------------------------------------------------------------------------
%-------------------------------------------------------------------------
% For incremental training input vectors and corresponding targets must be
% presented as sequences:
P = {[1;2],[2;1],[2;3],[3;1]}
T = {4,5,7,7}
%-------------------------------------------------------------------------
%-------------------------------------------------------------------------
% When using the adapt funciton in order to incrementally train a neural
% network where the inputs are presented as a cell array of sequential
% vectors, the weights are updated as each input is presented to the
% network.
% ------------------------------------------------------------------------
% When the inputs are presented as a matrix of concurrent vectors, the
% weights are updated only after all inputs are presented (batch mode).
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% Train network incrementally.
% Network outputs should remain zero since the learning rate is zero.
[net,a1,e1] = adapt(net,P,T)
% If we now set the learnig rate to 0.1 we may see how the network is
% adjusted as each input is presented:
net.inputWeights{1,1}.learnParam.lr = 0.1;
net.biases{1}.learnParam.lr = 0.1;
[net,a2,e2] = adapt(net,P,T)
[net,a3,e3] = adapt(net,P,T)
%-------------------------------------------------------------------------
% The first element of the output vector a2 is 0 (a2[1]=0), as it was with 
% zero learning rate, since no update is made until the first input is 
% presented to the network. The second input is different, since the 
% weights have been updated. The weghts continue to be modified as each
% error is computed. If the network is capable and the learning rate is set
% correctly, the error will eventually be driven to zero.
for t = 1:1:200
    [net,at,et] = adapt(net,P,T);
end;
at
et
% Show network internal parameters.
IW = net.IW{1,1}
b = net.b{1}
% The correct values for the weights and biases are IW = [2 1] and b = 0.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% In batch training weights and biases are updated only after all the
% inputs and corresponding targets are presented to the network.
% Batch training can be done using either the adapt or the train function.
% Train is generally the best option since it has access to more efficient
% training algorithms.
% Incremental training can only be done with adapt.
% Train can only perform batch training.
%-------------------------------------------------------------------------
%-------------------------------------------------------------------------
% Batch training using adapt.
% Network initialization.
net  = newlin([-1 1;-1 1],1,0,0.1)
net.IW{1,1} = [0 0];
net.b{1} = 0;
% Set training vectors and corresponding targets for batch training.
P = [1 2 2 3;2 1 3 1];
T = [4 5 7 7];
[net,a1,e1] = adapt(net,P,T)
% Note that the outputs a1 of the network are all zero, because the weights
% are not updated until all of the training set has been presented.
[net,a2,e2] = adapt(net,P,T)
[net,a3,e3] = adapt(net,P,T)
IW = net.IW{1,1}
b = net.b{1}
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% Batch training using train.
% The same bacth training can be performed using train. For this case, the 
% input vectors can be either placed in matrix of concurrent inputs or in a
% cell array of sequential vectors.
% Within train any cell array of sequential vectors is converted to a
% matrix of concurrent vectors.
%--------------------------------------------------------------------------
%-----------------------------Experiement 1--------------------------------
%--------------------------------------------------------------------------
clc
clear all
% Network initialization.
net  = newlin([-2 2;-2 2],1,0,0.1)
net = init(net);
% net.IW{1,1} = [0 0];
% net.b{1} = 0;
% Set training vectors and corresponding targets for batch training.
P = [1 2 2 3;2 1 3 1];
T = [4 5 7 7];
net.inputWeights{1,1}.learnParam.lr = 0.1;
net.biases{1}.learnParam.lr = 0.1;
net.trainParam.epochs = 20;
net = train(net,P,T)
Y1 = sim(net,P)
%--------------------------------------------------------------------------
%-----------------------------Experiement 2--------------------------------
%--------------------------------------------------------------------------
clc
clear all
% Network initialization.
net  = newlin([-2 2;-2 2],1,0,0.1)
net = init(net);
% net.IW{1,1} = [0 0];
% net.b{1} = 0;
% Set training vectors and corresponding targets for batch training.
P = [1 2 2 3;2 1 3 1];
T = [4 5 7 7];
net.inputWeights{1,1}.learnParam.lr = 0.01;
net.biases{1}.learnParam.lr = 0.01;
net.trainParam.epochs = 20;
net = train(net,P,T)
Y2 = sim(net,P)
%--------------------------------------------------------------------------
%-----------------------------Experiement 3--------------------------------
%--------------------------------------------------------------------------
clc
clear all
% Network initialization.
net  = newlin([-2 2;-2 2],1,0,0.1)
net = init(net);
% net.IW{1,1} = [0 0];
% net.b{1} = 0;
% Set training vectors and corresponding targets for batch training.
P = [1 2 2 3;2 1 3 1];
T = [4 5 7 7];
net.inputWeights{1,1}.learnParam.lr = 0.001;
net.biases{1}.learnParam.lr = 0.001;
net.trainParam.epochs = 20;
net = train(net,P,T)
Y = sim(net,P)