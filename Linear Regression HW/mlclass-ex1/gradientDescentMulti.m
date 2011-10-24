function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

% θj := θj - α 1/m m∑i=1 (hθ (x(i)) - y(i))xj(i)
	% n=size(X,2);
	% 	for j=1:n
	% 		sum1=0;
	% 		for i=1:m
	% 			prediction = 0;
	% 			for j_count=1:n
	% 				prediction = prediction + theta(j_count) + X(i,j_count);
	% 			end
	% 			sum1 = sum1 + (prediction - y(i))*(X(i,j));
	% 		end
	% 		theta(j) = theta(j) - alpha*(1/m)*sum1;
	% 	end


	theta = theta - alpha*(1/m)*((X'*X*theta)-(X'*y));




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
