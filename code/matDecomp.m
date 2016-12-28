function [v, U, S, W, Err] = matDecomp(M1, M2, k, maxItr, tol)
[m, n] = size(M1);

% Initialization
v = randn;
S = abs(randn(k,1));
U = randn(m, k);
[U, ~] = qr(U,0);
U = U(:,1:k);
W = randn(n, k);
[W, ~] = qr(W,0);
W = W(:,1:k);

%main iteration
for itr = 1:maxItr
    uOld = U; wOld = W; vOld = v; 
    
    for r = 1:k
        S_ = S;
        S_(r) = 0;
        A = U * diag(S_) * W';
        u=U(:,r); w=W(:,r);
        %U(:,r)=0; W(:,r)=0;
        
        %update equation
        u = (M1-v*M2-A) * w;
        denu = w' * w;
        u = u / denu;
        u = u / norm(u);
        
        w = u' * (M1-v*M2-A);
        denw = u' * u;
        w = w / denw;
        S(r) = norm(w);
        w = w / norm(w);
        
        U(:,r) = u;
        W(:,r) = w;
    end

    %In stead of doing the above alternating updates for svd, I use
    %MATLAB's svd. It is faster and my code does not converge for some
%     %unknown reason!!! Will need to resolve above later
%     [U,S,W] = svd( M1 - v*M2 , 'econ');
%     U = U(:,1:k);
%     W = W(:,1:k);
%     S = diag(S(1:k,1:k));

    res = M1 - U * diag(S) * W';
    v = sum(diag(res'*M2)) / norm(M2, 'fro')^2;
    
    Err(itr) = norm(M1 - v*M2 - U * diag(S) * W', 'fro');
    fprintf('Iter %d Error %f \n', itr, Err(itr));
%     Err2(itr) = (norm(U-uOld,'fro') + norm(W-wOld,'fro')) / 2;
    if Err(itr) < tol
        break;
    end
    
end



