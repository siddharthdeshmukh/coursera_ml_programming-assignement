function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
gradSize = size(theta)-1;
grad = zeros(size(gradSize));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

H = sigmoid(X * theta);

summation = sum(-1 * y' * log(H) - ((1-y')* log(1 -H)));

theta_zeroed_first = [0; theta(2:length(theta));];

J = (1/m )* summation + (lambda/(2 * m))* sum(theta_zeroed_first.^2) ; 

B = X' * (H -y) ;
A = (1/m) * B;
grad = A .+ (lambda/m)*theta_zeroed_first;




% =============================================================

end
