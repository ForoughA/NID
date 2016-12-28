function secOrdMomMu = SecMomEstWhite(trainData, W, mu, lengths)
% whitenedData = (trainData * W);
whitenedData = trainData ;
[trainNum, k] = size(whitenedData');
% whitenedMean = (mu' * W)';
whitenedMean = mu;
% k = size(R,2);
% M2 = zeros(vocabNum,k);

lengthsInv = 1./(lengths.*(lengths-1));
% lengthsInvMat = repmat(lengthsInv, [1,k]);

nzWhieInd = find(whitenedData);
nnzWhie = nnz(whitenedData);
[vocabId,docId] = ind2sub(size(whitenedData),nzWhieInd);
redDimWCVal = zeros(nnzWhie,1);
for i = 1:nnzWhie 
    redDimWCVal(i) = whitenedData(nzWhieInd(i)) * lengthsInv(docId(i));
end
redDimWC = sparse(docId,vocabId,redDimWCVal);
clear docId vocabId redDimWCVal lengths

% redDimWC = whitenedData .* lengthsInvMat;%For averaging purposes
% secOrdMom = whitenedData' * redDimWC - diag(whitenedData'*lengthsInv);
secOrdMom = whitenedData * redDimWC - diag(whitenedData*lengthsInv);
% secOrdMom = secOrdMom / trainNum;
E12mu = zeros(k, k*k);
for i = 1:k
    E12mu(:, (i-1)*k+1:i*k) = secOrdMom * whitenedMean(i);
end

muRep = repmat(whitenedMean, [1,trainNum]);
E1mu3 = redDimWC' * (kr(whitenedData, muRep))' - repmat(sum((kr(redDimWC', muRep)),2)',[k,1]);
Emu23 = whitenedMean * (sum(kr(redDimWC', whitenedData), 2))' - whitenedMean * (sum(kr(eye(size(redDimWC))', redDimWC'), 2))';
% E1mu3 = redDimWC' * (kr(whitenedData', muRep))' - lengthsInvMat' * (kr(whitenedData', muRep))';
% Emu23 = whitenedMean * (sum(kr(redDimWC', whitenedData'), 2))' - whitenedMean * (sum(kr(lengthsInvMat', whitenedData'), 2))';

secOrdMomMu = E12mu + E1mu3 + Emu23;
secOrdMomMu = secOrdMomMu / trainNum;
