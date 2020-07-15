%% Data process

load('face');                              

faces = reshape(X, [56,46,520]);
X = X ./ (ones(size(X,1),1) * sqrt(sum(X.*X)));                                                  
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
[Idx,Ctrs,SumD,D] = kmeans(train_X',32,'Replicates',100);
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
    for j=1:size(a,2)
        if a(i,j)==0
            P(i,j)=0;
        else
P(i,j)=train_l(a(i,j));
        end
    end
end  %�滻Ϊ��ʵ��ǩ
Ctrs=Ctrs';
%% perform @rank1
[min_a,index]=min(D,[],1);
for i=1:200
    for j=1:32
        d(i,j)=(test_X(:,i)-Ctrs(:,j))'*(test_X(:,i)-Ctrs(:,j));
    end
end %����codebook distance matrix
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
%% Accuracy
 matchcount = 0;
for k=1:numel(label)
    
        Predict = find(NP(k,:) ==label(k) ); 
     
        if ~isempty(Predict)
            matchcount = matchcount + numel(Predict);
        end
    
end
rate=matchcount/320;

% % c = [2, 3, 5, 7; 3, 5, 2, 8; 9, 5, 7, 8; 2, 2, 3, 9];
% c = P;
% % c = [4,8,7,15,12;7,9,17,14,10;6,9,12,8,7;6,7,14,6,10;6,9,12,10,6]; % end up with endless loop
% %function result = Hungarian(c)
% size = length(c);
% %pre-operation
% line = min((c'));
% line = repmat(line',1, size);
% c = c - line;
% column = min(c);
% column = repmat(column, size, 1);
% c = c - column;
% 
% %%
% [axis_line, axis_column] = find(c == 0);
% %C = [axis_line axis_column];
% result = [axis_line, axis_column];
% % [x, y] = meshgrid(temp_line, temp_column);
% % result =  [x(:), y(:)];
% %get the left position
% 
% %decode the matrix to three dimension
% %test = cell(size, 1);
% z = zeros(size, size);
% for i=1:size
%     x = find(result(:, 2) == i);
%     temp_size = length(x);
%     z(i, 1) = temp_size;
%     for j = 2:temp_size+1
%         test = result(x(j - 1),1);
%         z(i, j) = test;
%     end
% end
% 
% 
% %%
% %mark
% %flag = zeros(size, size);
% %this loop is aimed to finding the answer for sereral iteration
% tempresult = result;
% while true
%     %mark
%     markedLine = zeros(1, size);
%     markedColumn = zeros(1, size);
%     [tempAnswer, all] = fit(z, 1, (1:size), zeros(1, size), []);
%     left = setdiff((1:size), tempAnswer);
%     if ~isempty(all)
%         disp('all rolution');
%         for i = 1:length(all(:, 1))
%             fprintf('the %dth solution', i);
%             disp(all(i, :));
%         end
%         break;
%     end
%     %mark the row
%     %flag(left, :) = 1;
%     markedLine(left) = 1;
%     %this loop is aimed to find the possible zero
%     %find all marked lines
%     temp_marked = find(markedLine == 1);
%     for i = 1:length(temp_marked)
%         tmp = temp_marked(i);
%         while true
%            %%
%             %step1
%             %find the line(subscript)
%             tmp = find(tempresult(:, 1) == tmp);
%             erase = tmp;
%             %find the colum
%             tmp = tempresult(tmp, 2);
%             %find weather it is terminate
%             if isempty(tmp)
%                 break;
%             end
%             %mark the column
%             %flag(:,tmp) = 1;
%             markedColumn(tmp) = 1;
%             %erase the line
%             tempresult(erase, :) = [];
% 
%            %%
%             %step2
%             %find the line(subscript)
%             tmp = find(tempresult(:, 2) == tmp);
%             erase = tmp;
%             %find the line
%             tmp = tempresult(tmp, 1);
%             %find weather it is terminate
%             if isempty(tmp)
%                 break;
%             end
%             %mark the line
%             %flag(tmp, :) = 1;
%             markedLine(tmp) = 1;
%             %erase the column
%             tempresult(erase, :) = [];
%         end
%     end
% 
%     %%
%      %find the minimal number
%      line = markedLine;
%      column = not(markedColumn);
%      flag = (line')*column;
%      subscript = find(flag == 1);
%      min = findMin(c, subscript);
%      %add and sub
%      templine = find(markedLine == 1);
%      tempcolumn = find(markedColumn == 1);
%      c(templine, :) = c(templine, :) - min;
%      c(:, tempcolumn) = c(:, tempcolumn) + min;
%      temp = find(c(subscript) == 0);
%      temp = subscript(temp);
%      %for i = 1:length(temp)
%          y = fix((temp-1)/5) + 1;
%          x = temp - (y-1)*5;
%      %end
%      tempresult = [result; [x, y]];
%      %check the answer                                             
%      for i = 1:length(x)
%          z(x(i), 1) = z(x(i), 1) + 1;
%          z(x(i), z(x(i), 1)+1) = y(i); 
%      end
% end
% result = all;


% for i=1:32
%     b=[];
%     Pi=P(i,:);
%    B= Pi(Pi~=0);
%    c=mode(B);
%    for j=1:numel(label)
%    b=[b find(label(j)==c)];
%    end
%  
%    while (~isempty(b))
%         B= B(B~=c);
%         c=mode(B);
%    for j=1:numel(label)
%         b=find(label(j)==c);
%    end
%    end
%     label=[label; c];
% 
% end
% label=sort(label);
% find(Vec==u(2))
% %% Recognition rate 
%  matchcount = 0;
% for k=1:320
%     
%         if(idx(k) == train_l(k))
%      
%             matchcount = matchcount + 1;
%         end
%     
% end
% rate=matchcount/320;
% %�����ȡ150����
% X = [randn(50,2)+ones(50,2);randn(50,2)-ones(50,2);randn(50,2)+[ones(50,1),-ones(50,1)]];
% opts = statset('Display','final');
% 
% %����Kmeans����
% %X N*P�����ݾ���
% %Idx N*1������,�洢����ÿ����ľ�����
% %Ctrs K*P�ľ���,�洢����K����������λ��
% %SumD 1*K�ĺ�����,�洢����������е���������ĵ����֮��
% %D N*K�ľ��󣬴洢����ÿ�������������ĵľ���;
% 
% [Idx,Ctrs,SumD,D] = kmeans(X,3,'Replicates',3,'Options',opts);
% 
% %��������Ϊ1�ĵ㡣X(Idx==1,1),Ϊ��һ��������ĵ�һ�����ꣻX(Idx==1,2)Ϊ�ڶ���������ĵڶ�������
% plot(X(Idx==1,1),X(Idx==1,2),'r.','MarkerSize',14)
% hold on
% plot(X(Idx==2,1),X(Idx==2,2),'b.','MarkerSize',14)
% hold on
% plot(X(Idx==3,1),X(Idx==3,2),'g.','MarkerSize',14)
% 
% %����������ĵ�,kx��ʾ��Բ��
% plot(Ctrs(:,1),Ctrs(:,2),'kx','MarkerSize',14,'LineWidth',4)
% plot(Ctrs(:,1),Ctrs(:,2),'kx','MarkerSize',14,'LineWidth',4)
% plot(Ctrs(:,1),Ctrs(:,2),'kx','MarkerSize',14,'LineWidth',4)
% 
% legend('Cluster 1','Cluster 2','Cluster 3','Centroids','Location','NW')
% 
% 
% 
