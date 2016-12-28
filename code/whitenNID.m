function [W, UV, diagSigInv, v] = whitenNID(trainData, mean, lengths, k, maxItr, tol, rep)
vocabNum = size(trainData,2);
kPrime = 2*k;%randi([k+1, 2*k]);%Choose a k' greater than k
randomMat = randn(vocabNum, kPrime);%Form a vocabNum x k' random matrix
[M2R, M1R] = SecMomEstNID(trainData, mean, randomMat, lengths);
% [M2, M1] = SecMomEstNIDFull(trainData, mean, lengths);

err = 1000;
% rep = 20;
for i = 1:rep
    [vTmp, UTmp, ~, ~, errTmp] = matDecomp(M2R, M1R, k, maxItr, tol);%find the eigenvectors of (M2 - v*M1) * R and v
    if err(end) > errTmp(end)
        v = vTmp;
        U = UTmp;
        err = errTmp;
    end
end
% errf = 1000;
% for i = 1:rep
%     [vfTmp, UfTmp, ~, ~, errTmpf] = matDecomp(M2, M1, k, maxItr, tol);
%     if errf(end) > errTmpf(end)
%         vf = vfTmp;
%         Uf = UfTmp;
%         errf = errTmpf;
%     end
% end
% v = 0.0099;

%Choose the top k eigenvectors
% U = U(:,1:k);

[M2U, M1U] = SecMomEstNID(trainData, mean, U, lengths);
UM2U = U' * (M2U - v*M1U);

% [M2Uf, M1Uf] = SecMomEstNID(trainData, mean, Uf, lengths);
% UM2Uf = Uf' * (M2Uf - vf*M1Uf);

[V, sig, ~] = eig(UM2U);
diagSigInv = diag(1./sqrt(diag(abs(sig))));
UV = U * V;
W = UV * diagSigInv;



% [Vf, sigf, ~] = eig(UM2Uf);
% Wf = Uf * Vf * diag(1./sqrt(diag(abs(sigf))));
% W = V * diag(1./sqrt(diag(abs(sig))));
uW = pinv(W);

