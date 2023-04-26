function ReportTopNConnectedComponents(C,N,authors)

% This function reports the top N (measured by size) connected components 
% of the co-authorship network that are stored in cell array C. 
% The number of top N components and the initial authors' list are passed 
% input arguments to the function.

% Get the number of connected components.
components_num = length(C);
% Get the size of each connected component.
components_sizes = zeros(1,components_num);
for k = 1:1:components_num
    components_sizes(k) = length(C{k});
end;
% Sort connected components sizes in descending order.
[SortedComponentsSizes,SortedComponentsIndices] = sort(components_sizes,'descend');
% Get the top N connected components sizes and corresponding indices.
TopNComponentsSizes = SortedComponentsSizes(1:N);
TopNComponentsIndices = SortedComponentsIndices(1:N);

% Report Connected Components.
% Cycle through the top N connected components:
for n = 1:1:N
    component_index = TopNComponentsIndices(n);
    component_size = TopNComponentsSizes(n);
    component = C{component_index};
    fprintf('Component %d of size %d\n',component_index,component_size);
    % Cycle through the authors of each connected component:
    for m = 1:1:component_size
        author_index = component(m);
        author_firstname = authors(author_index,3);
        author_lastname = authors(author_index,2);
        fprintf('%d: %s %s\n',m,cell2mat(author_lastname),cell2mat(author_firstname));
    end;
end;

end

