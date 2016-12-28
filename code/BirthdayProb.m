clear

ppl = 10;
days = 365;

N = 10000;
success = zeros(1,N);
BirthDay = zeros(ppl,1);

for n = 1:N
    for p = 1:ppl
        BirthDay(p) = randi(days,1);
    end
    success(n) = (length(unique(BirthDay)) ~= length(BirthDay));
end

finalProb = sum(success) / N;