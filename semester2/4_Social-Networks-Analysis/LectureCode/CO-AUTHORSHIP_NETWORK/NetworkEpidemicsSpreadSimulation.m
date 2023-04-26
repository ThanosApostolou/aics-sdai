function [InfectionStatus] = NetworkEpidemicsSpreadSimulation(W,Po,P)

% This function simulates the spread of an epidemic within a network whose
% adjacency matrix W is given as an input argument. Po is the initial
% probality of a node to be infected at time t=0; P is the probability of
% the infection to spread along an edge from an infected node to an
% uninfected one.

% Get the number of nodes within the network.
N = size(W,1);
% All nodes are initially uninfected.
InfectionStatus = zeros(1,N);
% Initialize InfectedNodesFraction vector.
InfectedNodesFraction = [];
% Set network's infection status at time t=0.
for n = 1:1:N
    R = rand();
    if(R<=Po)
        InfectionStatus(n) = 1;
    end;
end;
converged = false;
t = 0;
while(~converged)
    t = t+1;
    % Store InfectionStatus before updating.
    PastInfectionStatus = InfectionStatus;
    % Store the currently infected and uninfected nodes.
    infected_nodes = find(InfectionStatus==1);
    uninfected_nodes = find(InfectionStatus==0);
    % For each uninfected node:
    for k = 1:1:length(uninfected_nodes)
        node = uninfected_nodes(k);
        neighbours = find(W(node,:)==1);
        infected_neighbours = intersect(neighbours,infected_nodes);
        % Each infected neighbours has a probability of P to spread the
        % infection.
        for l = 1:1:length(infected_neighbours)
            R = rand();
            if(R<=P)
                InfectionStatus(node) = 1;
            end;
        end;
    end;
    InfectedNodesFraction = [InfectedNodesFraction,sum(InfectionStatus) / N];
    if(sum(PastInfectionStatus-InfectionStatus)==0)
        converged = true;
    end;
end;

figure('Name','Infected Node Fraction Time Evolution');
time = [1:1:t];
plot(time,InfectedNodesFraction,'-*r','LineWidth',1.5');
xlabel('Time');
ylabel('Infected Nodes Fraction');
grid on

end

