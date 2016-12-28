function [secOrdMom, shiftTerm] = SecMomEstNID(trainData, mean, R, lengths)
trainNum = size(trainData, 1);
k = size(R,2);
% M2 = zeros(vocabNum,k);

lengthsInv = 1./(lengths.*(lengths-1));
lengthsInvMat = repmat(lengthsInv, [1,k]);
redDimWC = trainData * R;%reduced dimension word count matrix
redDimWC = redDimWC .* lengthsInvMat;%For averaging purposes
% secOrdMom = trainData' * redDimWC - diag(trainData'*lengthsInv)*R;
secOrdMom = trainData' * redDimWC - repmat((trainData'*lengthsInv), [1,k]).*R;
secOrdMom = secOrdMom / trainNum;

shiftTerm = mean * (mean' * R);