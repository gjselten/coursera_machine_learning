function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% find all X that belong to centroid 1
for k=1:K  % go over all centroids one by one
  nr_examples = 0;
  sum = zeros(n, 1);
  for i=1:m
    if (idx(i) == k) % if this datapoint matches this centroid we count it
      sum = sum + X(i, :)';         % add this datapoint to sum
      nr_examples++;                % +1 examples
    endif
  end
  centroids(k,:) = (sum/nr_examples)';
end






% =============================================================


end
