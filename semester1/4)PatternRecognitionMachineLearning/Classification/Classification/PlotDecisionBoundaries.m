function PlotDecisionBoundaries(R1,R2,net)

% This function plots the decision boundary between the classes C1 and C2
% that has been learned by the non-linear multi-layer perceptron stored in
% variable net.
% R1 and R2 are [2x2] matrices of the following forms:
%      |x11_min x11_max|          |x21_min x22_max|
% R1 = |x12_min x12_max| and R2 = |x22_min x22_max|

% Plot decision region.
% Reset dxy interval.
dxy = 0.01;
% Get minimum and maximum values for the x1 and x2 dimensions.
x1_min = min([R1(1,:),R2(1,:)]);
x1_max = max([R1(1,:),R2(1,:)]);
x2_min = min([R1(2,:),R2(2,:)]);
x2_max = max([R1(2,:),R2(2,:)]);
% Obtain the meshgrid surface upon which the decision region will be
% plotted.
x1 = [x1_min:dxy:x1_max];
x2 = [x2_min:dxy:x2_max];
[X1,X2] = meshgrid(x1,x2);
% Obtain vector versions for the meshgrid components.
X1 = X1(1:end);
X2 = X2(1:end);
GridPatterns = [X1',X2'];
GridClasses = sim(net,GridPatterns');
GridClasses = round(GridClasses);

Class1Indices = find(GridClasses==1);
Class2Indices = find(GridClasses==2);
Class1Patterns = GridPatterns(Class1Indices,:);
Class2Patterns = GridPatterns(Class2Indices,:);

figure('Name','Decision Boundaries Plot','NumberTitle','off')
hold on
plot(Class1Patterns(:,1),Class1Patterns(:,2),'+r');
plot(Class2Patterns(:,1),Class2Patterns(:,2),'og');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class1 Patterns','Class2 Patterns');
grid on
hold off



end

