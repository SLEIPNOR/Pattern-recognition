%% Data process

load('face');                              

faces = reshape(X, [56,46,520]);

num_trainImg = 8;
num_class = size(unique(l), 2);
trainIdx = [];
testIdx = [];
XX=[];
for i=1:num_class
	label = find(l == i);  %��ȡ��ǩ�е����У�10�ţ�ͼƬ������ǩ1-10��
% 	indice = randperm(numel(label)); %��Ԫ�ظ���Ϊ10�����ú���randperm��10������û��� 
%   trainIdx = label(indice(1:num_trainImg));               %���ſ��������׾���
% 	testIdx =  label(indice(num_trainImg+1:end));
% 	trainIdx = [trainIdx label(indice(1:num_trainImg))];    % 10������Ԫ��������ѡȡ1��8����C = A(B)��ƴ�Ӿ����γɰ�indince�����ȡÿ��label����8��ͼƬ��ѵ������
% 	testIdx = [testIdx label(indice(num_trainImg+1:end))];  % ���ѡȡ����ѵ������
trainIdx = [trainIdx label(1:8)];    
testIdx = [testIdx label(9:10)]; 
end                                                     
train_X = X(:, trainIdx);%��ȡѵ��ͼƬ
train_l = l(trainIdx);%��ȡѵ��label
test_X = X(:, testIdx);%��ȡ����ͼƬ
test_l = l(testIdx);%��ȡ����label

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
avg_face = mean(train_X,2); 		            % compute the average face ��ֵ����

%% Visualize of meanface
% figure(2)
% imshow(uint8(reshape(avg_face, [56,46])));
% %  imagesc(reshape(avg_face, 56, 46));
% 
% % title('Average Face')
%% Caculate eigenfaces AAT
sta_face= bsxfun(@minus,train_X,avg_face);          % subtract the mean face ��׼�����
% A=reshape(sta_face, [56,46,416]);
% imshow(uint8(A(:,:,1)));
% S = (sta_face*sta_face')/416; %Э�������
% [V, D] = eigs(S,416);
% eigenfaces = V;
%  D=diag(D);
% [ D,I]=sort(D,'descend');
% eigenfaces = eigenfaces ./ (ones(size(eigenfaces,1),1) * sqrt(sum(eigenfaces.*eigenfaces)));    %normalization ��һ��
%% Caculate eigenfaces ATA

% A=reshape(sta_face, [56,46,416]);
% imshow(uint8(A(:,:,1)));
Sl = (sta_face'*sta_face)/416; %ATA
[Vl, Dl] = eigs(Sl,415);
eigenfaces2 = sta_face*Vl;
 Dl=diag(Dl);
% [ D,I]=sort(D,'descend');
eigenfaces2 = eigenfaces2 ./ (ones(size(eigenfaces2,1),1) * sqrt(sum(eigenfaces2.*eigenfaces2)));    %normalization ��һ��
%% Visualize of eigenfaces 
% eigenfaces = reshape(eigenfaces, [56,46*40,1]);
% eigenfaces = mat2gray(eigenfaces);
% imshow(double(eigenfaces(:,:,1)));
%% Generate Transform axis train image
 espace=eigenfaces2'; %%ת�����ռ�����
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
distance = sqrt(bsxfun(@plus, bsxfun(@plus, AB, BB), AA')); %ŷʽ���� ת�÷�ǰ��˭ת�ú���˭���䣨���Ծ��󲻱䣩
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

 

                                    




