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
         
 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

y_eye = eye(num_labels);
y_matrix = y_eye(y,:);

a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones((size(z2,1)), 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);


J = ((1/m) * (sum(sum(-y_matrix .* log(a3) - ...
        (1-y_matrix) .* log(1-a3)))));

un_1 = sum(sum((Theta1(:,2:end)).^2));

un_2 = sum(sum((Theta2(:,2:end)).^2));

J = J + ((lambda / (2 * m)) * (un_1 + un_2));

d3 = a3 - y_matrix;
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

Delta1 = d2' * a1;
Delta2 = d3' * a2;

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_grad = Theta1_grad + ((lambda/m) * Theta1);
Theta2_grad = Theta2_grad + ((lambda/m) * Theta2);


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
