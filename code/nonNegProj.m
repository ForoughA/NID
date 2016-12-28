function P = nonNegProj(V)
% Boyd text book: Convex Optimization. Section 8.1.1 projection onto the 
% proper cone K for the special case of the non-negative orthant
P = zeros(size(V));
P(V>0) = V(V>0);