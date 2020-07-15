%% Data process

load('face');                              

% faces = reshape(X, [56,46,520]);

% X = X ./ (ones(size(X,1),1) * sqrt(sum(X.*X)));                                                    
train_X = X(:, 1:320);
train_l = l(1:320);
test_X = X(:, 321:520);
test_l = l(321:520);
ik=1;%rank
NR=[];
In=[];
InT=[];
APM=[];
ui=[];
ui8=[];
for g=1:200
nbins=51;
[N,edges] = histcounts(test_X(:,g),nbins,'BinLimits',[0,255]);
NR=[NR,N'];
end
train_X=NR;
for i=33:52
	label = find(test_l == i);  %抽取标签中的所有（10张）图片，（标签1-10）
    ui = [ui mean(NR(:, label(1:10)),2)];
end     
%% PCA computing average
avg_face = mean(train_X,2); 		            % compute the average face 均值矩阵

%% Caculate eigenfaces ATA
sta_face= bsxfun(@minus,train_X,avg_face);          % subtract the mean face 标准差矩阵
% A=reshape(sta_face, [56,46,416]);
% imshow(uint8(A(:,:,1)));
SL = (sta_face'*sta_face)/200; %协方差矩阵
[V, D] = eigs(SL,30);%  N-C
eigenfaces = sta_face*V;
%  D=diag(D);
% [ D,I]=sort(D,'descend');
eigenfaces = eigenfaces ./ (ones(size(eigenfaces,1),1) * sqrt(sum(eigenfaces.*eigenfaces)));    %normalization 归一化
%% Generate Transform axis train image
 espace=eigenfaces'; %%转换基空间生成
 ldtrain_X = espace*train_X; 
%  ldtrain_X8=mean(ldtrain_X(:,1:8),2);
%  ldtrain_X=eigenfaces*ldtrain_X;
%  faces = reshape(ldtrain_X, [56,46,416]);
% imshow(uint8(faces(:,:,1)));
%% LDA computation
ui= espace * ui; 
u= espace * avg_face; 
uimu=bsxfun(@minus,ui,u);
SB=10*(uimu*uimu');
for k=1:20
    for p=1:10
        ui8=[ui8 ui(:,k)];
    end
end
Sw=(ldtrain_X-ui8)*(ldtrain_X-ui8)';
R=Sw^(-1)*SB;
[w, A] = eigs(R,20);
w = w ./ (ones(size(w,1),1) * sqrt(sum(w.*w)));    %normalization 归一化
yn = w'* ldtrain_X; 
if nbins == 1
    rate=0.05;
end
if nbins ~=1
AB = -2 * (yn') * yn;      
BB = sum(yn .* yn);         
AA = sum(yn.* yn);   
distance = sqrt(bsxfun(@plus, bsxfun(@plus, AB, BB), AA'));

[B,I]  = mink(distance,ik+1,2);
for p=1:200
    Ir=I(p,1:ik+1);
     De = find(Ir == p); 
     Ir(De)=[];
     In=[In; test_l(Ir)];
end

%% Recognition rate 
 matchcount = 0;
for k=1:200
    
        label = find(In(k,1:ik) ==test_l(k) ); 
     
        if ~isempty(label)
            matchcount = matchcount + 1;
        end
    
end
rate=matchcount/200;
end
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
