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
Z=linkage(Y,'Ward');%%�Ա�Single:��̾��� Complete�������� Average��ƽ������ Centroid �����ľ��� Ward�����ƽ����
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
end %������ӻ����γɾ��󣬲��㱣֤ά����ͬ
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
%�滻Ϊ��ʵ��ǩ
%% label allocation algorithm
for i=1:32
 L = unique(P(i,:));
 NUM(i)=numel(L);
end %��ȡÿ�����е���������
NUM1=NUM;
for i=1:32
    [OV,Or]=min(NUM1);
    NUM1(Or)=NaN;
    NP(i,:)=P(Or,:);
end %�������а����������ӵ͵�������
for i=1:32 %ÿ��ȡmax������������Ϊ��ǩ
    b=[];
    Pi=NP(i,:);
   B= Pi(Pi~=0);
   c=mode(B);
   for j=1:numel(label)
   b=[b find(label(j)==c)];%����Ƿ�������ظ�
   end 
 
   if ~isempty(b) %�ظ�����ѭ����������
       while(1)
        b=[]; 
        B= B(B~=c);
        if isempty(B)
            c=0;
            break; %B��ֵ�˳�ѭ�� ��ֵ0
        else  
        c=mode(B);
        
   for j=1:numel(label)
        b=[b find(label(j)==c)];
   end
   if isempty(b) %���ظ��˳�ѭ��
       break;
   end
        end
       end   
       end
   
    label=[label; c]; %�γɱ�ǩ����

end
    NV=find(label==0);%Ѱ��0λ��
for i=1:numel(label)
  
    rest=find(label==i);
    if isempty(rest)
        Lrest=[Lrest i];
    end
end  %Ѱ��δ�����ֵ
for i=1:numel(NV)
    label(NV(i))=Lrest(i); %����δ��������0λ��
end
labe=sort(label);

d = codebook_distance_matrix(test_X, center, 32); %����codebook distance matrix

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
AB = -2 * (d') * d; % (ŷʽ������������ƶȶ�ѡһ��     
BB = sum(d .* d);         
AA = sum(d.* d);   
distance = sqrt(bsxfun(@plus, bsxfun(@plus, AB, BB), AA'));
% 
[KNNB,KNNI]  = mink(distance,rank+1,2);%ȡǰk+1��
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