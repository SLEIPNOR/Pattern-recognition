%% Data process

load('face');                              

% faces = reshape(X, [56,46,520]);

num_trainImg = 8;
num_class = size(unique(l), 2);
trainIdx = [];
testIdx = [];
ui = [];
ui8=[];
for i=1:num_class
	label = find(l == i);  %抽取标签中的所有（10张）图片，（标签1-10）
    trainIdx = [trainIdx label(1:num_trainImg)];    
    testIdx = [testIdx label(num_trainImg+1:end)]; 
    ui = [ui mean(X(:, label(1:num_trainImg)),2)];
end                                                     
train_X = X(:, trainIdx);%提取训练图片
train_l = l(trainIdx);%提取训练label
test_X = X(:, testIdx);%提取测试图片
test_l = l(testIdx);%提取测试label
%% PCA computing average
avg_face = mean(train_X,2); 		            % compute the average face 均值矩阵

%% Caculate eigenfaces ATA
sta_face= bsxfun(@minus,train_X,avg_face);          % subtract the mean face 标准差矩阵
% A=reshape(sta_face, [56,46,416]);
% imshow(uint8(A(:,:,1)));
SL = (sta_face'*sta_face)/416; %协方差矩阵
[V, D] = eigs(SL,364);%  N-C
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
SB=num_trainImg*(uimu*uimu');
for k=1:52
    for p=1:num_trainImg
        ui8=[ui8 ui(:,k)];
    end
end
Sw=(ldtrain_X-ui8)*(ldtrain_X-ui8)';
R=Sw^(-1)*SB;
[w, A] = eigs(R,364);
w = w ./ (ones(size(w,1),1) * sqrt(sum(w.*w)));    %normalization 归一化
%% Visualize of Fisherfaces 
% w=eigenfaces*w;
% w = reshape(w, [56,46*5,6]);
% imshow(mat2gray(w(:,:,1)));
%% Generate Transform axis train image
yn = w'* ldtrain_X; 
% ST=(bsxfun(@minus,ldtrain_X,u))*(bsxfun(@minus,ldtrain_X,u))'; 
% ST= w'* ST * w; 
%% Generate Transform axis test image
ynt = w'* espace * test_X;
%% Recgonization
AB = -2 * ynt' * yn;      
BB = sum(yn .* yn);         
AA = sum(ynt.* ynt);   
distance = sqrt(bsxfun(@plus, bsxfun(@plus, AB, BB), AA')); %欧式距离 
[e, index] = min(distance, [], 2);
%% Recognition rate 
 matchcount = 0;
for i=1:numel(index)
	predict = train_l(index(i));
	if predict == test_l(i)
		matchcount = matchcount + 1;
	end
end
rate=matchcount/numel(index);
fprintf('**************************************\n');
fprintf('accuracy: %0.3f%% \n', 100 * matchcount / numel(index));
fprintf('**************************************\n');
 