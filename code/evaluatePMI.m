function pmi = evaluatePMI(recBeta, n, testData)

%PMI score on top-n words 
%source: http://newport.eecs.uci.edu/anandkumar/pubs/AnandkumarValluvanStat13.pdf
[~, II] = sort(recBeta, 1, 'descend'); 
k = size(recBeta,2);
testNum = size(testData,1);

pmi = 0;
for h = 1:k
    for j=1:n
        for i=1:j-1
            Xi = II(i,h);
            Xj = II(j,h);
            Pi = sum(testData(:,Xi)>0) / testNum;
            Pj = sum(testData(:,Xj)>0) / testNum;
            Pij = sum(testData(:,Xi).*testData(:,Xj)>0) / testNum;
            
            if Pij~=0
                pmi = pmi + log(Pij/(Pi*Pj));
            end
        end
    end
end
pmi = pmi / (45*k);