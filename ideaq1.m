%% Data process

load('face');                              

faces = reshape(X, [56,46,520]);

num_trainImg = 8;
num_class = size(unique(l), 2);
trainIdx = [];
testIdx = [];
XX=[];
for i=1:num_class
	label = find(l == i);  %抽取标签中的所有（10张）图片，（标签1-10）
% 	indice = randperm(numel(label)); %（元素个数为10，并用函数randperm（10）随机置换） 
%   trainIdx = label(indice(1:num_trainImg));               %竖着看（矩阵套矩阵）
% 	testIdx =  label(indice(num_trainImg+1:end));
% 	trainIdx = [trainIdx label(indice(1:num_trainImg))];    % 10个打乱元素序列中选取1到8个，C = A(B)；拼接矩阵形成按indince随机抽取每个label类型8张图片的训练矩阵
% 	testIdx = [testIdx label(indice(num_trainImg+1:end))];  % 随机选取两个训练矩阵
trainIdx = [trainIdx label(1:8)];    
testIdx = [testIdx label(9:10)]; 
end                                                     
train_X = X(:, trainIdx);%提取训练图片
train_l = l(trainIdx);%提取训练label
test_X = X(:, testIdx);%提取测试图片
test_l = l(testIdx);%提取测试label

%% Visualize all faces
% fig = zeros(560,2392);
% for j = 1:52
%     for i = 10*j-8:10*j
%         im_o = faces(:,:,10*j-9);
%         im_t = faces(:,:,i);
%         if i == 10*j-8
%             im = [im_o;im_t];  
%         else
%             im = [im;im_t];
%         end  
%     end
%     fig(:,46*j-45:46*j) = im;
% end
% 
% figure(1)
% imshow(uint8(fig));
% title('All Faces')

%% PCA computing
disp('Computing Eigenfaces...');
% tic;
avg_face = mean(train_X,2); 		            % compute the average face 均值矩阵

%% Visualize of meanface
% figure(2)
% imshow(uint8(reshape(avg_face, [56,46])));
% %  imagesc(reshape(avg_face, 56, 46));
% 
% % title('Average Face')
%% Caculate eigenfaces AAT
sta_face= bsxfun(@minus,train_X,avg_face);          % subtract the mean face 标准差矩阵
% A=reshape(sta_face, [56,46,416]);
% imshow(uint8(A(:,:,1)));
% S = (sta_face*sta_face')/416; %协方差矩阵
% [V, D] = eigs(S,416);
% eigenfaces = V;
%  D=diag(D);
% [ D,I]=sort(D,'descend');
% eigenfaces = eigenfaces ./ (ones(size(eigenfaces,1),1) * sqrt(sum(eigenfaces.*eigenfaces)));    %normalization 归一化
%% Caculate eigenfaces ATA

% A=reshape(sta_face, [56,46,416]);
% imshow(uint8(A(:,:,1)));
Sl = (sta_face'*sta_face)/416; %ATA
[Vl, Dl] = eigs(Sl,415);
eigenfaces2 = sta_face*Vl;
 Dl=diag(Dl);
% [ D,I]=sort(D,'descend');
eigenfaces2 = eigenfaces2 ./ (ones(size(eigenfaces2,1),1) * sqrt(sum(eigenfaces2.*eigenfaces2)));    %normalization 归一化
%% Visualize of eigenfaces 
% eigenfaces = reshape(eigenfaces, [56,46*40,1]);
% eigenfaces = mat2gray(eigenfaces);
% imshow(double(eigenfaces(:,:,1)));
%% Generate Transform axis train image
 espace=eigenfaces2'; %%转换基空间生成
 ldtrain_X = espace*sta_face; 
%% Generate Transform axis test image
sta_tface = bsxfun(@minus, test_X, avg_face);
ldtest_X = espace * sta_tface;
% %% Reconstruction of a linear combination face
% R_face = avg_face + eigenfaces*ldtrain_X;
%% Visualization of a linear combination face
% R_face = reshape(R_face, [56,46,416]);
% imshow(uint8(R_face(:,:,1)));
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

 

                                    




