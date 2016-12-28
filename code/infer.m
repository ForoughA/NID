function [likelihood, converged] = infer(doc, alpha0, logBeta, tol, maxIter)

wordInd = find(doc);
wordCount = doc(doc ~= 0);
N = length(wordInd);
total = sum(wordCount);
k = size(logBeta, 2);
alpha = alpha0 / k * ones(1,k);

zeta = 1/k * ones(N,k);
h = alpha + total/k;

oldLikelihood = 0;
converged = zeros(1,maxIter);
converged(1) = 100;
iter = 1;

while ( converged(iter) > tol && iter <= maxIter )
    
    iter = iter + 1;
    for n = 1:N
        oldZeta = zeta(n,:);
        zeta(n,:) = logBeta(wordInd(n), :) + psi(h); %psi(h) computes the digamma function
        logSum = log(sum(exp(zeta(n,:))));
        zeta(n,:) = exp( zeta(n,:) - logSum );
        h = h + wordCount(n) * (zeta(n,:)-oldZeta);
%         h = alpha + wordCount(n) * (zeta(n,:));
    end
    
    likelihood = likelihoodEst(alpha0, alpha, logBeta, doc, h, zeta);
    converged(iter) = (oldLikelihood-likelihood)/oldLikelihood;
    oldLikelihood = likelihood;
end

converged(1) = [];