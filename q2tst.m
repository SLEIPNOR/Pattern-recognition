%% Data process

load('face');                              

faces = reshape(X, [56,46,520]);

num_trainImg = 8;
num_class = size(unique(l), 2);
trainIdx = [];
testIdx = [];
trainIdx1 = [];
trainIdx2 = [];
trainIdx3 = [];
trainIdx4 = [];
trainIdx123 = [];
trainIdx12=[];
for i=1:num_class
	label = find(l == i);  %抽取标签中的所有（10张）图片，（标签1-10）
% 	indice = randperm(numel(label)); %（元素个数为10，并用函数randperm（10）随机置换） 
%   trainIdx = label(indice(1:num_trainImg));               %竖着看（矩阵套矩阵）
% 	testIdx =  label(indice(num_trainImg+1:end));
% 	trainIdx = [trainIdx label(indice(1:num_trainImg))];    % 10个打乱元素序列中选取1到8个，C = A(B)；拼接矩阵形成按indince随机抽取每个label类型8张图片的训练矩阵
% 	testIdx = [testIdx label(indice(num_trainImg+1:end))];  % 随机选取两个训练矩阵
trainIdx = [trainIdx label(1:num_trainImg)];    
testIdx = [testIdx label(num_trainImg+1:end)]; 
end                                                     
train_X = X(:, trainIdx);%提取训练图片
train_l = l(trainIdx);%提取训练label
test_X = X(:, testIdx);%提取测试图片
test_l = l(testIdx);%提取测试label
avg_face = mean(train_X,2); 
sta_face= bsxfun(@minus,train_X,avg_face); 
% test_X=reshape(test_X, [56,46*8,13]);
% imshow(uint8(test_X(:,:,1)));
%% Splitting 1
for i=1:num_class
labelS = find(train_l == i);  
	
%   trainIdx = label(indice(1:num_trainImg));             
% 	testIdx =  label(indice(num_trainImg+1:end));
	trainIdx1 = [trainIdx1 labelS(1:2)];   	
end
train_X1 = train_X(:, trainIdx1);%提取第一组训练图片
train_l1 = train_l(trainIdx1);%提取第一组训练label
%% Splitting 2
for i=1:num_class
labelS = find(train_l == i);  
	
%   trainIdx = label(indice(1:num_trainImg));             
% 	testIdx =  label(indice(num_trainImg+1:end));
	trainIdx2 = [trainIdx2 labelS(3:4)];   	
end
train_X2 = train_X(:, trainIdx2);%提取第2组训练图片
train_l2 = train_l(trainIdx2);%提取第2组训练label
%% Splitting 3
for i=1:num_class
labelS = find(train_l == i);  
	
%   trainIdx = label(indice(1:num_trainImg));             
% 	testIdx =  label(indice(num_trainImg+1:end));
	trainIdx3 = [trainIdx3 labelS(5:6)];   	
end
train_X3 = train_X(:, trainIdx3);%提取第3组训练图片
train_l3 = train_l(trainIdx3);%提取第3组训练label
%% Splitting 4
for i=1:num_class
labelS = find(train_l == i);  
	
%   trainIdx = label(indice(1:num_trainImg));             
% 	testIdx =  label(indice(num_trainImg+1:end));
	trainIdx4 = [trainIdx4 labelS(7:8)];   	
end
train_X4 = train_X(:, trainIdx4);%提取第4组训练图片
train_l4 = train_l(trainIdx4);%提取第4组训练label


%% Splitting 123
for i=1:num_class
labelS = find(train_l == i);  
	
%   trainIdx = label(indice(1:num_trainImg));             
% 	testIdx =  label(indice(num_trainImg+1:end));
	trainIdx123 = [trainIdx123 labelS(1:6)];   	
end
train_X123 = train_X(:, trainIdx123);%提取第4组训练图片
train_l123 = train_l(trainIdx123);%提取第4组训练label
%% Splitting 12
for i=1:num_class
labelS = find(train_l == i);  
	
%   trainIdx = label(indice(1:num_trainImg));             
% 	testIdx =  label(indice(num_trainImg+1:end));
	trainIdx12 = [trainIdx12 labelS(1:4)];   	
end
train_X12 = train_X(:, trainIdx12);%提取第4组训练图片
train_l12 = train_l(trainIdx12);%提取第4组训练label

%% Computing S1+S2=S12
avg_face1 = mean(train_X1,2); 	
avg_face2 = mean(train_X2,2); 	
avg_face12 = (104*avg_face1+104*avg_face2)/208;
% imshow(uint8(reshape(avg_face3, [56,46])));
sta_face1 = bsxfun(@minus,train_X1,avg_face1);  
sta_face2 = bsxfun(@minus,train_X2,avg_face2); 
sta_face12 = bsxfun(@minus,train_X12,avg_face12); 
S1 = (sta_face1*sta_face1')/104;
S2 = (sta_face2*sta_face2')/104;
S1l= (sta_face1'*sta_face1)/104;
S2l= (sta_face2'*sta_face2)/104;
S12=(104/208)*S1+(104/208)*S2+((104*104)/(208*208))*(avg_face1-avg_face2)*(avg_face1-avg_face2)';
[P1, D1] = eigs(S1l,103);
[P2, D2] = eigs(S2l,103);
P1 = sta_face1*P1;
P2 = sta_face2*P2;
u12 = avg_face1-avg_face2;
%% Gram-Schmidt orthogonalization
O12=[P1,P2,u12];
 h12 = GS(O12);
% h = h ./ (ones(size(h,1),1) * sqrt(sum(h.*h)));
C12 = h12'*S12* h12;
[R12, DC12] = eigs(C12,207);
P12 = h12 *R12;
%% Generate Transform axis train image
 espace=P12'; %%转换基空间生成
 ldtrain_X12 = espace*sta_face12; 
%% Generate Transform axis test image
sta_tface = bsxfun(@minus, test_X, avg_face12);
ldtest_X = espace * sta_tface;
%% Recgonization
AB = -2 * ldtest_X' * ldtrain_X12;      
BB = sum(ldtrain_X12 .* ldtrain_X12);         
AA = sum(ldtest_X.* ldtest_X);   
distance = sqrt(bsxfun(@plus, bsxfun(@plus, AB, BB), AA')); %欧式距离 转置放前面谁转置横向谁不变（测试矩阵不变）
[e, index] = min(distance, [], 2);
%% Recognition rate 
 matchcount = 0;
for i=1:numel(index)
	predict = train_l12(index(i));
	if predict == test_l(i)
		matchcount = matchcount + 1;
	end
end
rate12=matchcount/numel(index);
%% Computing S12+S3=S123
	
avg_face3 = mean(train_X3,2); 	
avg_face123 = (208*avg_face12+104*avg_face3)/312;
sta_face3 = bsxfun(@minus,train_X3,avg_face3); 
S3 = (sta_face3*sta_face3')/104;
S3l= (sta_face3'*sta_face3)/104;
S123=(208/312)*S12+(104/312)*S3+((208*104)/(312*312))*(avg_face12-avg_face3)*(avg_face12-avg_face3)';
[P3, D3] = eigs(S3l,103);
P3 = sta_face3*P3;
u123 = avg_face12-avg_face3;
sta_face123 = bsxfun(@minus,train_X123,avg_face123); 
%% Gram-Schmidt orthogonalization
O123=[P12,P3,u123];
 h123 = GS(O123);
% h = h ./ (ones(size(h,1),1) * sqrt(sum(h.*h)));
C123 = h123'*S123* h123;
[R123, DC123] = eigs(C123,311);
P123 = h123 *R123;
%% Generate Transform axis train image
 espace=P123'; %%转换基空间生成
 ldtrain_X123 = espace*sta_face123; 
%% Generate Transform axis test image
sta_tface = bsxfun(@minus, test_X, avg_face123);
ldtest_X = espace * sta_tface;
%% Recgonization
AB = -2 * ldtest_X' * ldtrain_X123;      
BB = sum(ldtrain_X123 .* ldtrain_X123);         
AA = sum(ldtest_X.* ldtest_X);   
distance = sqrt(bsxfun(@plus, bsxfun(@plus, AB, BB), AA')); %欧式距离 转置放前面谁转置横向谁不变（测试矩阵不变）
[e, index] = min(distance, [], 2);
%% Recognition rate 
 matchcount = 0;
for i=1:numel(index)
	predict = train_l123(index(i));
	if predict == test_l(i)
		matchcount = matchcount + 1;
	end
end
rate123=matchcount/numel(index);
%% Computing S123+S4=S1234
	
avg_face4 = mean(train_X4,2); 	
avg_face1234 = (312*avg_face123+104*avg_face4)/416;
sta_face4 = bsxfun(@minus,train_X4,avg_face4); 
S4 = (sta_face4*sta_face4')/104;
S4l = (sta_face4'*sta_face4)/104;
S1234=(312/416)*S123+(104/416)*S4+((312*104)/(416*416))*(avg_face123-avg_face4)*(avg_face123-avg_face4)';

[P4, D4] = eigs(S4l,103);
P4=sta_face4*P4;
u1234 = avg_face123-avg_face4;
%% Gram-Schmidt orthogonalization
O1234=[P123,P4,u1234];
 h1234 = GS(O1234);
% h = h ./ (ones(size(h,1),1) * sqrt(sum(h.*h)));
C1234 = h1234'*S1234* h1234;
[R1234, DC1234] = eigs(C1234,415);
P1234 = h1234 *R1234;

%% Generate Transform axis train image
 espace=P1234'; %%转换基空间生成
 ldtrain_X = espace*sta_face; 
%% Generate Transform axis test image
sta_tface = bsxfun(@minus, test_X, avg_face);
ldtest_X = espace * sta_tface;

% %% Reconstruction of a linear combination face
% R_face = avg_face + P1234*ldtrain_X;
% % Visualization of a linear combination face
% R_face = reshape(R_face, [56,46,416]);
% imshow(uint8(R_face(:,:,9)));

%% Recgonization
AB = -2 * ldtest_X' * ldtrain_X;      
BB = sum(ldtrain_X .* ldtrain_X);         
AA = sum(ldtest_X.* ldtest_X);   
distance = sqrt(bsxfun(@plus, bsxfun(@plus, AB, BB), AA')); %欧式距离 转置放前面谁转置横向谁不变（测试矩阵不变）
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
 
 


% P3 = reshape(P3, [56,46,101]);
% 
% imshow(mat2gray(P3(:,:,71)));

% %% Batch PCA
% [P1234b, DC1234b] = eigs(S1234,415);






% %% Visualize all faces
% % fig = zeros(560,2392);
% % for j = 1:52
% %     for i = 10*j-8:10*j
% %         im_o = faces(:,:,10*j-9);
% %         im_t = faces(:,:,i);
% %         if i == 10*j-8
% %             im = [im_o;im_t];  
% %         else
% %             im = [im;im_t];
% %         end  
% %     end
% %     fig(:,46*j-45:46*j) = im;
% % end
% % 
% % figure(1)
% % imshow(uint8(fig));
% % title('All Faces')
% 
% %% PCA computing
% disp('Computing Eigenfaces...');
% % tic;
% avg_face = mean(train_X,2); 		            % compute the average face 均值矩阵
% 
% %% Visualize of meanface
% % figure(2)
% % imshow(uint8(reshape(avg_face, [56,46])));
% % %  imagesc(reshape(avg_face, 56, 46));
% % 
% % % title('Average Face')
% %% Caculate eigenfaces
% sta_face= bsxfun(@minus,train_X,avg_face);          % subtract the mean face 标准差矩阵
% % A=reshape(sta_face, [56,46,416]);
% % imshow(uint8(A(:,:,1)));
% S = (sta_face*sta_face'); %协方差矩阵
% [V, D] = eigs(S,400);
% eigenfaces = V;
% %  D=diag(D);
% % [ D,I]=sort(D,'descend');
% eigenfaces = eigenfaces ./ (ones(size(eigenfaces,1),1) * sqrt(sum(eigenfaces.*eigenfaces)));    %normalization 归一化
% %% Visualize of eigenfaces 
% % eigenfaces = reshape(eigenfaces, [56,46*40,1]);
% % eigenfaces = mat2gray(eigenfaces);
% % imshow(double(eigenfaces(:,:,1)));
% %% Generate Transform axis train image
%  espace=eigenfaces'; %%转换基空间生成
%  ldtrain_X = espace*sta_face; 
% %% Generate Transform axis test image
% sta_tface = bsxfun(@minus, test_X, avg_face);
% ldtest_X = espace * sta_tface;
% %% Reconstruction of a linear combination face
% R_face = avg_face + eigenfaces*ldtrain_X;
% %% Visualization of a linear combination face
% % R_face = reshape(R_face, [56,46,416]);
% % imshow(uint8(R_face(:,:,1)));
% %% Recgonization
% AB = -2 * ldtest_X' * ldtrain_X;      
% BB = sum(ldtrain_X .* ldtrain_X);         
% AA = sum(ldtest_X.* ldtest_X);   
% distance = sqrt(bsxfun(@plus, bsxfun(@plus, AB, BB), AA')); %欧式距离 转置放前面谁转置横向谁不变（测试矩阵不变）
% [e, index] = min(distance, [], 2);
% %% Recognition rate 
%  matchcount = 0;
% for i=1:numel(index)
% 	predict = train_l(index(i));
% 	if predict == test_l(i)
% 		matchcount = matchcount + 1;
% 	end
% end
% rate=matchcount/numel(index);
% fprintf('**************************************\n');
% fprintf('accuracy: %0.3f%% \n', 100 * matchcount / numel(index));
% fprintf('**************************************\n');
%  

                                    

