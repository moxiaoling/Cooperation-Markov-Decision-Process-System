function [Pi00,Pi11,V0,V1,V]= Two_V_convergence_test()
%write by Mo Xiaoling
%The probability transfer matrix and reward value matrix of the two agents are uniform
%During the interaction of two agents, the corresponding value function of the two agents will converge

Ns0=3;   %State number
Na0=2;   %Action number
S0=[1,2,3]; %State set
A0=[1,2];   %Action set
Ns1=3;   %State number
Na1=2;   %Action number
S1=[1,2,3]; %State set
A1=[1,2];   %Action set

%Probability transfer functionT(s,a,s'), rewrite toT(s,s',a)
%T0(:,:,1)=[0.2  0.3  0.5
%          0.7  0.1  0.2
%          0.3  0.1  0.6];
%T0(:,:,2)=[0.4  0.5  0.1
%          0.1  0.4  0.5
%          0.3  0.3  0.4];
      
T0(:,:,1)=[0.5  0  0.5
          0.7  0.1  0.2
          0.4  0  0.6];
T0(:,:,2)=[0   0    1
          0   0.95  0.05
          0.3  0.3  0.4];
%Reward functionR(s,a,s'). rewrite to R(s,s',a)      
%R0(:,:,1)=[1  1  1
%          2  1  1
%          1  1  2];
%R0(:,:,2)=[1  1  1
%          1  1  1
%          -1  1  1];
R0(:,:,1)=[1  0  1
          5  1  1
          1  0  1];
R0(:,:,2)=[0  0  1
          0  1  1
          -1  1  1];

%%T represents the tensor product of the probability transfer of two agent states, the probability of the transfer of Agent1
% after the transfer of Agent0, and T00 represents the tensor product of the behavior
T00(:,:,1)=kron(T0(:,:,1),T0(:,:,1));%Agent0 execute a0,Agent1 execute a0
T00(:,:,2)=kron(T0(:,:,1),T0(:,:,2));%Agent0 execute a0,Agent1 execute a1
T00(:,:,3)=kron(T0(:,:,2),T0(:,:,1));%Agent0 execute a1,Agent1 execute a0
T00(:,:,4)=kron(T0(:,:,2),T0(:,:,2));%Agent0 execute a1,Agent1 execute a1
%%R represents the tensor product of the probability transfer of two agent states; after the transfer of Agent0,
% the reward value of the transfer of Agent1; R01 represents the reward value of the behavior
R00(:,:,1)=kron(R0(:,:,1),R0(:,:,1));
R00(:,:,2)=kron(R0(:,:,1),R0(:,:,2));
R00(:,:,3)=kron(R0(:,:,2),R0(:,:,1));
R00(:,:,4)=kron(R0(:,:,2),R0(:,:,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:4
    Essr(:,:,i)=(T00(:,:,i)*R00(:,:,i)');%%The value expectations of each state under different strategies
end
for i=1:9
    Essr1(i,1)=Essr(i,i,1);% pi0(1,1)=pi1(1,1)£¬expectations
    Essr2(i,1)=Essr(i,i,2);% pi0(1,2)
    Essr3(i,1)=Essr(i,i,3);% pi1(2,1)
    Essr4(i,1)=Essr(i,i,4);% pi0(2,2)=pi1(2,2)
end
  
gamma=0.95;%discount factorgamma;
alpha=0.3;%equilibrium parameter£¬V0 equal alpha,V1 equal 1-alpha
Max=300;%The largest number of iterations
V0=ones(Ns0*Ns1,Max);%Produces an Ns0 *Ns1 dimensions full zero square matrix 
V1=ones(Ns1*Ns0,Max);
V=ones(Ns1*Ns0,Max);

%Pi00=ceil(2*rand(1,3));%Randomly generate a set of policies
%Pi11=ceil(2*rand(1,3));%Randomly generate a set of policies
Pi00=[1,2,2];%Agent0 will execute actions
Pi11=[2,1,2];%Agent1 will execute actions
%(Pi0,Pi1)=(Pi00(x),Pi11(y))
for M=1:Max-1%%Circular 100 times
    for s=1:Ns0
        for t=1:Ns1
            
            a0=Pi00(s);%%The initial strategy for agent0
            
            a1=Pi11(t);%%The initial strategy for agent1
            if(a0==1 && a1==1)
                V0((s-1)*Ns1+t,M+1)=Essr1((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,1)*gamma*V(:,M);%Value = probability * (reward value + discount factor * value), V function of agent0 
                V1((s-1)*Ns1+t,M+1)=Essr1((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,1)*gamma*V(:,M);%Agengt1 execute a1,Agent0 execute a1
                V((s-1)*Ns1+t,M+1)=alpha* V0((s-1)*Ns1+t,M+1)+(1-alpha)*V1((s-1)*Ns1+t,M+1);
            elseif(a0==1 && a1==2)
                V0((s-1)*Ns1+t,M+1)=Essr2((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,2)*gamma*V(:,M);%Agengt1 execute a1,Agent0 execute a2
                V1((s-1)*Ns1+t,M+1)=Essr3((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,3)*gamma*V(:,M);
                V((s-1)*Ns1+t,M+1)=alpha* V0((s-1)*Ns1+t,M+1)+(1-alpha)*V1((s-1)*Ns1+t,M+1);
            elseif (a0==2 && a1==1)
                V1((s-1)*Ns1+t,M+1)=Essr3((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,3)*gamma*V(:,M);%Agengt1 execute a2,Agent0 execute a1
                V0((s-1)*Ns1+t,M+1)=Essr2((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,2)*gamma*V(:,M);
                V((s-1)*Ns1+t,M+1)=alpha* V0((s-1)*Ns1+t,M+1)+(1-alpha)*V1((s-1)*Ns1+t,M+1);
            else
                V0((s-1)*Ns1+t,M+1)=Essr4((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,4)*gamma*V(:,M);%Agengt1 execute a2,Agent0 execute a2
                V1((s-1)*Ns1+t,M+1)=Essr4((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,4)*gamma*V(:,M);
                V((s-1)*Ns1+t,M+1)=alpha* V0((s-1)*Ns1+t,M+1)+(1-alpha)*V1((s-1)*Ns1+t,M+1);
            end
        end
    end
end


hold off

s=1:Ns0*Ns1;
x=1:Max;
  plot(x,V(s,x),'-')%The convergence of the V function of Agent0
  hold on
%  plot(x,V(1,x+1), k,V(2,x+1), k,V(3,x+1), k,V(4,x+1), k,V(5,x+1), k,V(6,x+1), k,V(7,x+1), k,V(8,x+1), k,V(9,x+1))
%  plot(x,V1(s,x),'-.g')%The convergence of the V function of Agent1
  xlabel('The number of iterations');
  ylabel('Expected reward value');
  hold on

  title('Convergence graph of V(s) under strategy Pi')

Pi00
Pi11
V00=V0(:,Max)'%The final value of V0
V11=V1(:,Max)'%The final value of V1
Total00 =sum(V0(:,Max))%The sum of the final V0 values at the end of the iteration
Total11 =sum(V1(:,Max))%The sum of the final V1 values at the end of the iteration

end


