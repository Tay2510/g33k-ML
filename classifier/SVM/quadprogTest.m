% Primal
x = randn(40,2);
d =[ones(20,1); -ones(20,1)]; x = x + d * [.5 .5];
H = diag([0 1 1 zeros(1,80)]);
gamma =1;
f= [zeros(43,1); gamma*ones(40,1)];
Aeq = [d x.*(d*[1 1]) -eye(40) eye(40)];
beq = ones(40,1);
A =zeros(1,83);
b = 0;
lb = [-inf*ones(3,1); zeros(80,1)];
ub = [inf*ones(83,1)];
[w,fval] = quadprog(gamma*H,f,A,b,Aeq,beq,lb,ub);

% Dual
xn = x.* (d*[1 1]);
k= xn*xn';
gamma =1;
f= -ones(40,1); Aeq = d';
beq = 0;
A =zeros(1,40); b = 0;
lb = [zeros(40,1)];
ub = [gamma*ones(40,1)];
[alpha,fvala] = quadprog(k,f,A,b,Aeq,beq,lb,ub);