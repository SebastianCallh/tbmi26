X = [2 5 2 2;
     1 3 3 5];
Y = [1, 2, 1, 3];

% K neighbours
k = 3;
% New point to classify
x = [1;
     2];

% Make distance map
A = x*ones(1,size(X,2)) - X;
D = zeros(1,size(X,2));
for column = (1:size(X,2))
    D(column) = sqrt(sum(A(:,column).^2));
end

[N I] = sort(D);

mode(Y(I(1:k)));

