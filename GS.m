function [v] = GS(A)
v(:,1)=A(:,1)/norm(A(:,1));
Num=0;%��������
[Ahang,Alie]=size(A);  %������к���
 for j=2:Alie
     sum(:,1)=zeros(1,Ahang); %��ʱ���� �洢��������
     for i=1:j-1
         h(i,j)=A(:,j)'*v(:,i);
          sum(:,1) =sum(:,1)+h(i,j)*v(:,i);
             Num=Num+1;
     end
     v(:,j)=A(:,j)-sum(:,1);
     v(:,j)=v(:,j)/norm(v(:,j));
 end
end
