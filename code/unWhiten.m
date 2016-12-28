function [alpha, beta] = unWhiten(U1, U2, U3, S ,W, UV, diagSigInv, alpha0)
k = size(U1,2);
n = size(W,1);
alpha = 1 ./ (S.^2);
% alpha = alpha / sum(alpha);
alpha = alpha * (alpha0*(alpha0+1)*(alpha0+2)) / 2;
SigInv = diag(diagSigInv);
SigInv( abs(SigInv) < 1e-6 ) = 0;
SigInv( abs(SigInv) > 1e-6 ) = 1;
diagSig = diag( SigInv ./ diag(diagSigInv) );
Winv = UV * diagSig;
Winv2 = pinv(W');
betaPrime = zeros(n,3*k);
for i = 1:k
    betaPrime(:,i) = S(i) * Winv * U1(:,i);
    betaPrime(:,i) = betaPrime(:,i) / sum(abs(betaPrime(:,i)));
%     betaPrime(:,k+i) = S(i) * Winv * U2(:,i);
%     betaPrime(:,2*k+i) = S(i) * Winv * U3(:,i);
end

beta = zeros(n,k);
for i = 1:k
    Ppos = nonNegProj(betaPrime(:,i));
    errPos = norm(Ppos - betaPrime(:,i));
    Pneg = nonNegProj(-betaPrime(:,i));
    errNeg = norm(Pneg + betaPrime(:,i));
    if errPos > errNeg
        beta(:,i) = Pneg;
    else
        beta(:,i) = Ppos;
    end
    beta(:,i) = beta(:,i) / sum(abs(beta(:,i)));
end

