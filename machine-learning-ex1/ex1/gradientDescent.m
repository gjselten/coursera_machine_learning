function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCost) and gradient here.
    %

    % THETA 0
    sum = 0;
    for i=1:m,
      expected = y(i);
      predicted = theta(1) + theta(2) * X(i,2);
      sum = sum + (predicted - expected);
    end
    theta_zero_subtract = alpha * (1 / m) * sum;
    theta_one = theta(1) - theta_zero_subtract;
    % theta(1) = theta(1) - theta_zero_subtract;

    % THETA 1
    sum = 0;
    for i=1:m,
      expected = y(i);
      predicted = theta(1) + theta(2) * X(i,2);
      sum = sum + ((predicted - expected) * X(i, 2));
    end
    theta_one_subtract = alpha * (1 / m) * sum;
    theta_two = theta(2) - theta_one_subtract;
    % theta(2) = theta(2) - theta_one_subtract;

    % update thetas simultaneous
    theta(1) = theta_one;
    theta(2) = theta_two;

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
