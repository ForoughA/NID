function [U1, U2, U3, S, v1, v2, Err] = tenDecomp(T1, T2, T3, k, maxItr, ninititr, tol, ninit)

    [n1, n2, n3] = size(T1);

    % Initialization
    v1 = randn;
    v2 = randn;

    % initialization by Robust Tensor Power Method (modified for non-symmetric tensors)
    U01 = zeros(n1,k);
    U02 = zeros(n2,k);
    U03 = zeros(n3,k);
    S0 = zeros(k,1);
    TE = T1 - v1*T2 - v2*T3;
    for i=1:k 
        tU1 = zeros(n1,ninit);
        tU2 = zeros(n2,ninit);
        tU3 = zeros(n3,ninit);
        tS  = zeros(ninit,1);
        for init=1:ninit
            [tU1(:,init), tU2(:,init), tU3(:,init)] = RTPM(TE-CPcomp(S0,U01,U02,U03), ninititr);  
            tU1(:,init) = tU1(:,init)./norm(tU1(:,init));
            tU2(:,init) = tU2(:,init)./norm(tU2(:,init));
            tU3(:,init) = tU3(:,init)./norm(tU3(:,init));
            tS(init) = TenProj(TE-CPcomp(S0,U01,U02,U03),tU1(:,init),tU2(:,init),tU3(:,init) );
        end
        [~, I] = max(tS);   
        U01(:,i) = tU1(:,I)/norm(tU1(:,I));
        U02(:,i) = tU2(:,I)/norm(tU2(:,I));
        U03(:,i) = tU3(:,I)/norm(tU3(:,I));
        S0(i) = TenProj(TE-CPcomp(S0,U01,U02,U03),U01(:,i),U02(:,i),U03(:,i));
    end
    
    U1 = U01; U2 = U02; U3 = U03; 
    S = S0; 
    
    for itr = 1:maxItr
        Uold1 = U1; Uold2 = U2; Uold3 = U3;
        Sold = S;
        
        for r=1:k
            S_ = S;
            S_(r)=0;
            A = CPcomp(S_,U1,U2,U3);
            u1=U1(:,r);u2=U2(:,r);u3=U3(:,r);
            U1(:,r)=0;U2(:,r)=0;U3(:,r)=0;
            den1=zeros(n1,1); den2=zeros(n2,1);
            s = S(r); 
            
            % Update
            TE = T1 - v1*T2 - v2*T3;
            for i3=1:n3
                U1(:,r) = U1(:,r) + u3(i3)*(TE(:,:,i3)-A(:,:,i3))*u2;
                den1    = den1    + u3(i3)^2*ones(n1,n2)*(u2.*u2);
            end
            u1 = U1(:,r)./den1;
            u1 = u1/norm(u1);
            
            for i3=1:n3
                U2(:,r) = U2(:,r) + u3(i3)*(TE(:,:,i3)-A(:,:,i3))'*u1;
                den2    = den2    + u3(i3)^2*ones(n2,n1)*(u1.*u1);
            end
            u2 = U2(:,r)./den2;
            u2 = u2/norm(u2);
            
            for i3=1:n3
                U3(i3,r) = ( u1'*(TE(:,:,i3)-A(:,:,i3))*u2 ) /( (u1.*u1)'*ones(n1,n2)*(u2.*u2) );
            end
            
            U1(:,r) = u1;
            U2(:,r) = u2;
            S(r)    = norm(U3(:,r)); 
            U3(:,r) = U3(:,r)/norm(U3(:,r)); 
            
        end
        
        res = T1 - CPcomp(S,U1,U2,U3);
        
        res1 = res - v2*T3;
        v1 = sum(sum(sum(T2 .* res1))) / sum(sum(sum(T2.^2)));
        
        res2 = res - v1*T2;
        v2 = sum(sum(sum(T3 .* res2))) / sum(sum(sum(T3.^2)));
        
        Err(itr) = sum( sum( sum( (res - v1*T2 - v2*T3).^2 ))) / sum( sum( sum( T1 )));
        if Err(itr) < tol
            break;
        end
    end
end



function T = CPcomp(S,u1,u2,u3)
   n1 = size(u1,1);n2 = size(u2,1);n3 = size(u3,1);
   r = min([length(S) size(u1,2) size(u2,2) size(u3,2)]);
   T = zeros(n1,n2,n3);
   for i=1:n3
        T(:,:,i) = u1*diag(u3(i,:).*S(:)')*u2';
   end
end

function M = TenProj(T,u1,u2,u3)
   n1=size(u1,1);n2=size(u2,1);n3=size(u3,1);
   r1=size(u1,2);r2=size(u2,2);r3=size(u3,2);
   M =zeros(r1,r2,r3);
   for i=1:r3
       A = zeros(n1,n2);
       for j=1:n3
           A = A+T(:,:,j)*u3(j,i);
       end
       M(:,:,i) = u1'*A*u2;
   end
end

function [u1, u2, u3] = RTPM(T, mxitr)
    n1=size(T, 1);n2=size(T, 2);n3=size(T, 3);
    uinit=randn(n1,1);u1=uinit/norm(uinit);
    uinit=randn(n2,1);u2=uinit/norm(uinit);
    uinit=randn(n3,1);u3=uinit/norm(uinit);
    for itr=1:mxitr
        vv1=zeros(n1,1);vv2=zeros(n2,1);v3=zeros(n3,1);
        for i3=1:n3
            v3(i3)=u1'*T(:, :, i3)*u2; 
            vv1 = vv1 + u3(i3)*T(:, :, i3)*u2;
            vv2 = vv2 + u3(i3)*T(:, :, i3)'*u1;
        end
        u10 = u1;
        u1 = vv1/norm(vv1);
        u20 = u2;
        u2 = vv2/norm(vv2);
        u30 = u3;
        u3 = v3/norm(v3);
        if(norm(u10-u1)+norm(u20-u2)+norm(u30-u3)<1e-7) break; end
    end
end
