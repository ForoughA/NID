clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hyper parameters
k = 10;
alpha0 = 0.01;
dataType = 1;%set dataType to 0 for c++ indexing and 1 for matlab
maxItr = 1000;
tol = 1e-3;
ninititr = 20;%number of initial iterations of tensor decomposition
ninit = 5;%number of initializations of tensor decomposition
repeats = 3;%number of repeats for choosing the best decomposition
dataPath = '../data/'; % This is the path to the data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load data: data can be loaded from .txt or .m
% data should be in the sparse bag of words format available in UCI Machine
% Learning Repository. Each row of data is of the following format:
% docID vocabID wordCount
% Where docID is the integer document ID, vocabID is the integer vocabulary 
% index in the dictionary and wordCount is the number of times vocabID is
% repeated in docID. 
%
% For loading data from .txt uncomment the following:
%%%%%%%%%%%%%%%%%%%%%
% filenameTrain = '../trainData.txt';
% filenameTest = '../testData.txt';
% [tmp1, docID, tmp2, vocabID, tmp3, counts, tmp4] = textread(filenameTrain,'%c %d %c %d %c %f %c');
% finalDocID = docID;
% finalVocabID = vocabID;
% finalCounts = counts;
% clear tmp1 docID tmp2 vocabID tmp3 counts tmp4;
% 
% [tmp1, docID, tmp2, vocabID, tmp3, counts, tmp4] = textread(filenameTest,'%c %d %c %d %c %f %c');
% finalDocIDTest = docID;
% finalVocabIDTest = vocabID;
% finalCountsTest = counts;
% 
% if dataType==0
%     trainData = sparse(finalDocID+1,finalVocabID+1,finalCounts);
%     testData = sparse(finalDocIDTest+1,finalVocabIDTest+1,finalCountsTest);
% elseif dataType==1
%     trainData = sparse(finalDocID,finalVocabID,finalCounts);
%     testData = sparse(finalDocIDTest,finalVocabIDTest,finalCountsTest);
% end
%%%%%%%%%%%%%%%%%%%%%
%
% For loading data from .mat uncomment the following:
% File should be saved in a sparse matrix whose rows are the documents and
% columns are vocabularies in the dictionary. Each matrix entry is the 
% load('../trainData.mat');
% load('../testData.mat');
%
%%%%%%%%%%%%%%%%%%%%%
fprintf('------------Reading Data------------ \n')
load([dataPath,'nyTimesTrain.mat']);
load([dataPath,'nyTimesTest.mat']);

% load('/Users/Forough/Desktop/datasets/nyTimes/nyTimesTrainLarge.mat');
% load('/Users/Forough/Desktop/datasets/nyTimes/nyTimesTestSmall.mat');

%%% whitening
fprintf('------------Whitening------------ \n')
lengths = sum(trainData,2);
smallDocs = find(full(lengths) < 3);
trainData(smallDocs,:) = [];
lengths(smallDocs) = [];
trainNum = size(trainData,1);
vocabNum = size(trainData,2);
lengthInv = 1./lengths;
mean = trainData'*lengthInv / trainNum;

[Wnid, UWnid, diagSigInv, vhat] = whitenNID(trainData, mean, lengths, k, maxItr, tol, repeats);
whitenedData = (trainData * Wnid)';
whitenedMean = (mean' * Wnid)';

%%% Third moment estimation %%% matricization %%%
fprintf('------------Moment Estimation------------ \n')
[T1, T2, T3] = ThrdMomEstNID(whitenedData, whitenedMean, Wnid, lengths);

%%% decomposition
fprintf('------------Decomposition------------ \n')
ErrFinal = 1000;
for i = 1:repeats
    [U1, U2, U3, S, v1hat, v2hat, Errhat] = tenDecomp(T1, T2, T3, k, maxItr, ninititr, 1e-4, ninit);
    if Errhat(end) < ErrFinal(end)
        U1Final = U1;
        U2Final = U2;
        U3Final = U3;
        SFinal = S;
        v1Final = v1hat;
        v2Final = v2hat;
        ErrFinal = Errhat;
    end
    fprintf('tenDecomp Err: %f \n', ErrFinal(end));
end

%%% unwhitening
fprintf('------------Unwhitening------------ \n')
[recAlpha, recBeta] = unWhiten(U1Final, U2Final, U3Final, SFinal, Wnid, UWnid, diagSigInv, alpha0);
recBetaNorm = recBeta ./ (repmat(sum(recBeta,1),[vocabNum,1]));

%%% Inference
fprintf('------------Inference------------ \n')
logBeta = log(recBetaNorm + eps);
llNID = 0;
maxItr = 10;
for testDoc = 1:size(testData,1)
    [likelihood, converged] = infer(testData(testDoc,:), alpha0, logBeta, 1e-5, maxItr);
    llNID = llNID + likelihood;
end

perpNID = exp(-llNID/sum(sum(testData)));%Perplexity
PMI = evaluatePMI(recBetaNorm, 10, testData);%PMI
