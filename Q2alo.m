  %% Data process

load('face');                              

faces = reshape(X, [56,46,520]);
% X = X ./ (ones(size(X,1),1) * sqrt(sum(X.*X)));                                                  
train_X = X(:, 1:320);
train_l = l(1:320);
test_X = X(:, 321:520);
test_l = l(321:520);
a=[];
P=[];
NP=[];
NUM=[];
label=[];
Lrest=[];
center=[];
rank=1;
In=[];
Y=pdist(train_X','euclidean');
Y=squareform(Y);
Z=linkage(Y,'Ward');%%对比Single:最短距离 Complete：最大距离 Average：平均距离 Centroid ：重心距离 Ward：离差平方和
dendrogram(Z);
c=cophenet(Z,Y);
T=cluster(Z,'maxclust',32);
Idx=T;
u=unique(Idx');
for i=1:32
A =find(Idx'==u(i));
 if isempty(a)
         a=[a;A];  
else
l=max([size(a,2),length(A)]);
A=[A , zeros(1,l-length(A))];
a=[a , zeros(i-1,l-size(a,2))];
a=[a;A];
end
end %聚类可视化，形成矩阵，补零保证维度相同
for i=1:32
       
    b = nonzeros(a(i,:));
    b=b';
     for j=1:size(b,2)
    
   center(:,i) = mean(train_X(:,b(1:j)),2); 
        end
     
end
    for i=1:32
        for j=1:size(a,2)
            if a(i,j)==0
                P(i,j)=0;
            else
    P(i,j)=train_l(a(i,j));
            end
        end
    end
%替换为真实标签
%% label allocation algorithm
for i=1:32
 L = unique(P(i,:));
 NUM(i)=numel(L);
end %提取每个类中的种类数量
NUM1=NUM;
for i=1:32
    [OV,Or]=min(NUM1);
    NUM1(Or)=NaN;
    NP(i,:)=P(Or,:);
end %将矩阵行按种类数量从低到高排序
for i=1:32 %每行取max数量的种类作为标签
    b=[];
    Pi=NP(i,:);
   B= Pi(Pi~=0);
   c=mode(B);
   for j=1:numel(label)
   b=[b find(label(j)==c)];%检查是否发生类别重复
   end 
 
   if ~isempty(b) %重复进入循环持续迭代
       while(1)
        b=[]; 
        B= B(B~=c);
        if isempty(B)
            c=0;
            break; %B空值退出循环 赋值0
        else  
        c=mode(B);
        
   for j=1:numel(label)
        b=[b find(label(j)==c)];
   end
   if isempty(b) %无重复退出循环
       break;
   end
        end
       end   
       end
   
    label=[label; c]; %形成标签矩阵

end
    NV=find(label==0);%寻找0位置
for i=1:numel(label)
  
    rest=find(label==i);
    if isempty(rest)
        Lrest=[Lrest i];
    end
end  %寻找未分配的值
for i=1:numel(NV)
    label(NV(i))=Lrest(i); %分配未分配的类给0位置
end
labe=sort(label);

d = codebook_distance_matrix(test_X, center, 32); %计算codebook distance matrix

%% Accuracy
 matchcount = 0;
for k=1:numel(label)
    
        Predict = find(NP(k,:) ==label(k) ); 
     
        if ~isempty(Predict)
            matchcount = matchcount + numel(Predict);
        end
    
end
rateT=matchcount/320;


%% perform @rank1
% %% problem a)
% 11111111111111111111111111111111111111111KNN
d=d';
AB = -2 * (d') * d; % (欧式距离和余弦相似度二选一）     
BB = sum(d .* d);         
AA = sum(d.* d);   
distance = sqrt(bsxfun(@plus, bsxfun(@plus, AB, BB), AA'));
% 
[KNNB,KNNI]  = mink(distance,rank+1,2);%取前k+1项
for p=1:200
    Ir=KNNI(p,1:rank+1);
     De = find(Ir == p); 
     Ir(De)=[];
     In=[In; test_l(Ir)];
end
%% Recognition rate 
 matchcount = 0;
for k=1:200
    
        label = find(In(k,1:rank) == test_l(k)); 
     
        if ~isempty(label)
            matchcount = matchcount + 1;
        end
    
end
rate=matchcount/200;