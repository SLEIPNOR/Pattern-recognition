clear;
load("face");

% A=reshape(X,[56,46,520]);
% i=1;
% while(i<10)
% C(i)=uint8(A(:,1:46,i));
% % D=uint8(A(:,1:46,2));
% % E=uint8(A(:,1:46,3));
% i=i+1;
% end
% 
% % F=cat(2,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10);
% imshow(uint8(C(i)(:,:,1)));
A=reshape(X,[56,46,520]);
% C=uint8(A(:,1:46,1));
imshow(mat2gray(A(:,:,9)));

% Averface=mean(X,2);
% %  Averface=reshape(Averface,[56,46]);
% % imshow(uint8(Averface(:,:)));
% sub=bsxfun(@minus,X,Averface);
% standardface=std(X,0,2);


% num_class = size(unique(l), 2);
% trainIdx = [];
% testIdx = [];




