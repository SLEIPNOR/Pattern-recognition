function y = gram_schmidt(u,num)
[m,n] = size(u);
if(m<n)
    error('��С���У��޷����㣬��ת�ú���������');
    return
end
k=num+1;
x = rand(m,m-n);
t = zeros(m,1);
y = [u,x];
y(:,1) = y(:,1)/norm(y(:,1));
for j = 1:m-k+1
    for i = 1:k-1
        t = t+y(:,i)'*y(:,k)*y(:,i);
    end
    p1 = y(:,k) - t;
    p1 = p1/norm(p1);
    y(:,k) = p1;
    k = k+1;
    t = zeros(m,1);
end
end