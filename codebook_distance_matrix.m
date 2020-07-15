function d = codebook_distance_matrix(test_X, Ctrs, centers)
for i=1:size(test_X,2)
    for j=1:centers
        d(i,j)=(test_X(:,i)-Ctrs(:,j))'*(test_X(:,i)-Ctrs(:,j));
    end
end