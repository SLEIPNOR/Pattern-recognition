%% Data process(using NCA)

load('face');                              

faces = reshape(X, [56,46,520]);

X = X ./ (ones(size(X,1),1) * sqrt(sum(X.*X)));                                                  
train_X = X(:, 1:320);
train_l = l(1:320);
test_X = X(:, 321:520);
test_l = l(321:520);
v=1;
In=[];
InT=[];
% [mappedX, mapping]=nca(train_X', train_l', 50, 0.001);

for i=1:200
    for j=1:200
MA(i,j)=(test_X(:,i)-test_X(:,j))'*(mapping.M)*(mapping.M)'*(test_X(:,i)-test_X(:,j));
    end
end
% AB = -2 * (test_X') * test_X;      
% BB = sum(test_X .* test_X);         
% AA = sum(test_X.* test_X);   
% distance = sqrt(bsxfun(@plus, bsxfun(@plus, AB, BB), AA'));
[B,I]  = mink(MA,v+1,2);
for p=1:200
    Ir=I(p,1:v+1);
     De = find(Ir == p); 
     Ir(De)=[];
     In=[In; test_l(Ir)];
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
% % %% AP caculation
% % i=199;
% % [BT,IT]  = mink(MA,i+1,2);
% % for p=1:200
% %     IrT=IT(p,1:i+1);
% %      DeT = find(IrT == p); 
% %      IrT(DeT)=[];
% %      InT=[InT; test_l(IrT)];
% % end
% % for p=1:200
% %    SAP=0;
% %     rank = find(InT(p,1:199) == test_l(p)); 
% %      
% %     for h=1:numel(rank)
% %         Pr(p,h)=h/rank(h);
% %         RC(p,h)=h/numel(rank);
% %     end
% % end
% % for j=1:200
% % for h=0:10
% %     RCAP=h/10;
% %     NewM = find(RC(j,:) >= RCAP);
% %     pinterp(j,h+1) = max(Pr(j,NewM(:)));
% % end
% % end
% % AP=mean(pinterp,2);
% % mAP=mean(AP);
