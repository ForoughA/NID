function P = evaluateProb(nid, nij, h, alpha, beta)

k = size(beta,2);
term1 = evaluateNID(alpha,h,k);
term2 = prod(h.^nid);
term3 = prod(prod(A.^nij));

P = term1 * term2 * term3;