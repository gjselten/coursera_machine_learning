function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% X contains 5000 x 400, which are 5000 training examples with 400 features each (the pixels)
% Theta1: the weights to go from the first layer (input layer) to second layer (hidden layer). 25x401
% Theta2: the weights to go from first layer to third layer (output layer) 10x26
% p contains 5000 x 1. The expected outcome is index that corresponds to a digit class.
%
% The second layer has 25 units
% The third layer has 10 units, corresponding to the 10 digit classes.


% go from input layer to hidden layer
Xtemp = [ones(m, 1) X];           % add ones to X
layer_1_z = Xtemp * Theta1';      % calculate Z for hidden layer
X2 = sigmoid(layer_1_z);          % calculate H for hidden layer. Called it 'X2', as it serves as input for the next layer.

% go from hidden layer to output layer
m2 = size(X2, 1);                    % get size of X2
Xtemp2 = [ones(m2, 1) X2];           % add ones to X2
layer_2_z = Xtemp2 * Theta2';        % calculate Z for hidden layer
h = sigmoid(layer_2_z);              % calculate H for output layer

% h contains a 5000 x 10 matrix. every row contains 10 elements. Every element
% is the probably that the input is this output (eg first element = 1, etc.)
% the highest number is the best match and therefore our prediction.
[max_value, index_max_value] = max(h(1,:));
[max_value, index_max_value] = max(h');
p = index_max_value;


% =========================================================================


end
