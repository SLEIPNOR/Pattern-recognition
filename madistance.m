%% Data process

load('face');                              

faces = reshape(X, [56,46,520]);
MaM=[];
X = X ./ (ones(size(X,1),1) * sqrt(sum(X.*X)));                                                  
train_X = X(:, 1:320);
train_l = l(1:320);
test_X = X(:, 321:520);
test_l = l(321:520);
v=1;%rank
In=[];
InT=[];
%% PCA computing average
avg_face = mean(train_X,2); 		            % compute the average face 均值矩阵

%% Caculate eigenfaces ATA
sta_face= bsxfun(@minus,train_X,avg_face);          % subtract the mean face 标准差矩阵
% A=reshape(sta_face, [56,46,416]);
% imshow(uint8(A(:,:,1)));
SL = (sta_face'*sta_face)/200; %协方差矩阵
[V, D] = eigs(SL,128);%  N-C 降维按题目要求选择维度 16,32,64,128,256
eigenfaces = sta_face*V;
%  D=diag(D);
% [ D,I]=sort(D,'descend');
eigenfaces = eigenfaces ./ (ones(size(eigenfaces,1),1) * sqrt(sum(eigenfaces.*eigenfaces)));    %normalization 归一化
%% Generate Transform axis train image
 espace=eigenfaces'; %%转换基空间生成
 ldtrain_X = espace*train_X; 
 ldtest_X= espace*test_X; 

u=mean(ldtrain_X,2);
sta_face= bsxfun(@minus,ldtrain_X,u);%Slove the cov matrix between imagie instead of dim
S=(sta_face*sta_face')/320;
S1in=inv(S);
for i=1:200
    for j=1:200
MA(i,j)=(ldtest_X(:,i)-ldtest_X(:,j))'*S1in*(ldtest_X(:,i)-ldtest_X(:,j));
    end
end
[B,I]  = mink(MA,v+1,2);
for p=1:200
    Ir=I(p,1:v+1);
     De = find(Ir == p); 
     Ir(De)=[];
     In=[In; test_l(Ir)];%rank 矩阵
end
%% Recognition rate 
 matchcount = 0;
for k=1:200
    
        label = find(In(k,1:v) == test_l(k)); 
     
        if ~isempty(label)
            matchcount = matchcount + 1;
        end
    
end
rate=matchcount/200;
%% AP caculation
i=199;
[BT,IT]  = mink(MA,i+1,2);
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


