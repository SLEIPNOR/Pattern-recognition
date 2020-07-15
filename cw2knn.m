 %% Data process

load('face');                              

faces = reshape(X, [56,46,520]);

X = X ./ (ones(size(X,1),1) * sqrt(sum(X.*X)));                                                  
train_X = X(:, 1:320);
train_l = l(1:320);
test_X = X(:, 321:520);
test_l = l(321:520);
i=1;
cosine=[];
In=[];
AP=[];
APM=[];
InT=[];
%% cos distance similarity(必须单位化向量)
for j=1:200
    for k=1:200
cosine(j,k)=1-(test_X(:,j)'*test_X(:,k));
    end
end
distance=cosine;
[B,I]  = mink(distance,i+1,2);%取前k+1项

% % 11111111111111111111111111111111111111111KNN
% AB = -2 * (test_X') * test_X; % (欧式距离和余弦相似度二选一）     
% BB = sum(test_X .* test_X);         
% AA = sum(test_X.* test_X);   
% distance = sqrt(bsxfun(@plus, bsxfun(@plus, AB, BB), AA'));
% 
% [B,I]  = mink(distance,i+1,2);%取前k+1项
for p=1:200
    Ir=I(p,1:i+1);
     De = find(Ir == p); 
     Ir(De)=[];
     In=[In; test_l(Ir)];
end
%% Recognition rate 
 matchcount = 0;
for k=1:200
    
        label = find(In(k,1:i) == test_l(k)); 
     
        if ~isempty(label)
            matchcount = matchcount + 1;
        end
    
end
rate=matchcount/200;
%% AP caculation
i=199;
[BT,IT]  = mink(distance,i+1,2);
for p=1:200
    IrT=IT(p,1:i+1);
     DeT = find(IrT == p); 
     IrT(DeT)=[];
     InT=[InT; test_l(IrT)];
end
for p=1:200
   SAP=0;
    rank = find(InT(p,1:199) == test_l(p)); 
     
    for h=1:numel(rank)
        Pr(p,h)=h/rank(h);
        RC(p,h)=h/numel(rank);
    end
end
for j=1:200
for h=0:10
    RCAP=h/10;
    NewM = find(RC(j,:) >= RCAP);
    pinterp(j,h+1) = max(Pr(j,NewM(:)));
end
end
AP=mean(pinterp,2);
mAP=mean(AP);





% cls = predict(kNNClassifier, X(:, 321:520)');
% A=cls';
% 
% %% Recognition rate 
%  matchcount = 0;
% for i=1:200
% 	predict = A(i);
% 	if predict == test_l(i)
% 		matchcount = matchcount + 1;
% 	end
% end
% rate=matchcount/200;
