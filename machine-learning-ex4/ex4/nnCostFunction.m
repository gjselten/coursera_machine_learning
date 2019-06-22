function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% STEP 1: CALCULATE H
% note this is similar to ex3 predict.m

% go from input layer to hidden layer
Xtemp = [ones(m, 1) X];           % add ones to X
layer_1_z = Xtemp * Theta1';      % calculate Z for hidden layer
X2 = sigmoid(layer_1_z);          % calculate H for hidden layer. Called it 'X2', as it serves as input for the next layer.

% go from hidden layer to output layer
m2 = size(X2, 1);                    % get size of X2
Xtemp2 = [ones(m2, 1) X2];           % add ones to X2
layer_2_z = Xtemp2 * Theta2';        % calculate Z for hidden layer
h = sigmoid(layer_2_z);              % calculate H for output layer

% STEP 2: transform y from 5000*1 to 5000*10
% - at this point 'y' is simply stating what class the answer is (eg 1,2,3 etc)
% - instead, we want y to be a vector of 10, where everything is 0, exept the right class.
%   eg 7 = 0 0 0 0 0 0 1 0 0 0
Y = zeros(size(y),num_labels);
for i=1:size(y)
  Y(i,y(i)) = 1;
end

% STEP 3: calculate J
J = (1 / m) * sum(sum((-1 * Y .* log(h) - ((1 - Y) .* log(1  - h)))));

% STEP 4: regularize
reg = lambda /(2 * m) * (sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)));
J = J + reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

accumulated_delta_1 = 0;
accumulated_delta_2 = 0;
for i=1:m
  % step 1 is forward pass (orward propagation). calculate all activations.
  % note, we could also have used the previously calculated values, but
  % following the instructions here seems cleaner.
  % a3 = h(i,:)
  % a2 = Xtemp2(i,:)

  a1 = Xtemp(i,:);
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [1 a2];
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);

  % step 2: delta for output layer
  % we can simply find this by subtracing the actual value from the h(=a3)
  d3 = a3 - Y(i,:);

  % step 3: delta for hidden layer
  % this is a bit harder but just following the formulas.
  temp = Theta2' * d3';
  d2 = temp(2:end)' .* sigmoidGradient(z2);

  % step 4: accumulate deltas
  accumulated_delta_1 = accumulated_delta_1 + d2' * a1;
  accumulated_delta_2 = accumulated_delta_2 +d3' * a2;
end

% step 5: obtain unregularized gradient
Theta1_grad = (1/m)*accumulated_delta_1;
Theta2_grad = (1/m)*accumulated_delta_2;

regTheta1 = Theta1;
regTheta1(:,1) = 0;
regTheta2 = Theta2;
regTheta2(:,1) = 0;

Theta1_grad = (1/m) * accumulated_delta_1 + ( lambda / m * regTheta1 );
Theta2_grad = (1/m) * accumulated_delta_2 + ( lambda / m * regTheta2 );

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
