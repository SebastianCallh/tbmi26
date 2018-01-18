function [ acc ] = calcAccuracy( cM )
    acc = sum(diag(cM))/sum(sum(cM));
end

