function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp1=theta(1);
temp2=theta(2);
for iter = 1:num_iters 
    temp1=theta(1)- (alpha*(1/m)*sum(X*theta-y));
    temp2=theta(2)-(alpha*(1/m)*sum((X*theta-y).*(X(:,2))));
    theta=[temp1 ; temp2];


    J_history(iter) = computeCost(X, y, theta);

end

end
