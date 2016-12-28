function likelihood = likelihoodEst(alpha0, alpha, logBeta, doc, h, zeta)

terms = zeros(5,1);

normH = zeros(size(h))';
logGamma = zeros(size(h));
for i = 1:size(logBeta,2)
    normH = (psi(h) - psi(sum(h)))';
    if h(i)>0
        logGamma(i) = gammaln(h(i));
    else
        logGamma(i) = log(gamma(h(i)));
    end
end
terms(1) = gammaln(alpha0) - sum(gammaln(alpha)) + ((alpha-1) * normH);
terms(2) = sum(zeta*normH);
terms(3) = sum(zeta*(doc*logBeta)');
terms(4) = -gammaln(sum(h)) + sum(logGamma) - ((h-1) * normH);
terms(5) = sum(sum(zeta .* log(zeta + eps)));

likelihood = sum(terms);