function d = softmax_proba(test_X, Ctrs)
sigma = 0.1;
for i = 1:size(test_X)
    temp_test = repmat(test_X(:,i),1,size(Ctrs,2));
    d(i,:) = 1/exp(-sum((temp_test - Ctrs).^2,1)/(2*sigma^2))/sum(exp(-sum((temp_test - Ctrs).^2,1)/(2*sigma^2)));
end

