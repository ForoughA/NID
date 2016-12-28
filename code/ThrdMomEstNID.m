function [T1, T2, T3] = ThrdMomEstNID(trainData, mean, W, lengths)
trainNum = size(trainData,2);
k = size(W,2);
whitenedData = trainData;%this is if we have whitened teh data outside the function
whitenedMean = mean;
% whitenedData = (trainData * W)';%This is if we haven't whitened the data
% outside the function.
% whitenedMean = (mean' * W)';

lengthsInv = 1 ./ (lengths .* (lengths-1) .* (lengths-2));
lengthsInvMat = repmat(lengthsInv', [k,1]);

%Sec 6. Discussion Eq (8),
%Tensor decomposition for learning latent variable models, Anima's paper
%First term: c \otimes c \otimes c
term1 = (lengthsInvMat .* whitenedData) * (kr(whitenedData, whitenedData))';
%second term: e_i \otimes e_i \otimes e_i
term2Val = (whitenedData * lengthsInv);
indTmp = (1:k)';
term2Ind = [indTmp, (indTmp-1)*k+indTmp];
term2 = sparse(term2Ind(:,1), term2Ind(:,2), term2Val);
term2 = full(term2);
% 3rd, 4rth, 5th terms
term3 = zeros(k,k*k);% e_i \otimes e_i \otimes e_j
term4 = zeros(k,k*k);% e_i \otimes e_j \otimes e_i
term5 = zeros(k,k*k);% e_i \otimes e_j \otimes e_j
for i = 1:k
    tmpVal = whitenedData * (whitenedData(i,:).*lengthsInv')';
    term3(:, (i-1)*k+1:i*k) = diag(tmpVal);
    term4(i, (i-1)*k+1:i*k) = tmpVal';
    term5(:, (i-1)*k+i) = tmpVal; 
end
thrdOrdMom = term1 + 2*term2 - term3 - term4 - term5;
thrdOrdMom = thrdOrdMom / trainNum;

%Can be made more efficient by feeding in the whitened data to both
%thrdMomEstDown and secMomEstWhite
secOrdMomMu = SecMomEstWhite(whitenedData, W, mean, lengths);

T1 = reshape(thrdOrdMom, [k,k,k]);
T2 = reshape(secOrdMomMu, [k,k,k]);
T3 = reshape(whitenedMean * (kr(whitenedMean, whitenedMean))', [k,k,k]);











