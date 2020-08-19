function [GH,V]= Two_compare_V_Pi_test()
%write by Mo Xiaoling
%The probability transfer matrix and reward value matrix of the two agents are identical
%Two agents, three states each, two actions, a total of 2*2*2 *2*2 *2 = 64 strategies
%Lo lists the strategies that make the value function optimal for three states of two agents in 64 strategies
%clear
Ns0=3;   %State number
Na0=2;   %Action number
S0=[1,2,3]; %State set
A0=[1,2];   %Action set
Ns1=3;   %State number
Na1=2;   %Action number
S1=[1,2,3]; %State set
A1=[1,2];   %Action set

%Probability transfer function T(s,a,s'), rewrite to T(s,s',a)
T0(:,:,1)=[0.5  0  0.5
          0.7  0.1  0.2
          0.4  0  0.6];
T0(:,:,2)=[0   0    1
          0   0.95  0.05
          0.3  0.3  0.4];
%T0(:,:,1)=[1/3  1/3  1/3
%          1/3  1/3  1/3
%          1/3  1/3  1/3];
%T0(:,:,2)=[1/3  1/3  1/3
%          1/3  1/3  1/3
%          1/3  1/3  1/3];
%Reward function R(s,a,s'). rewrite to R(s,s',a)     
R0(:,:,1)=[1  0  1
          5  1  1
          1  0  1];
R0(:,:,2)=[0  0  1
          0  1  1
          -1  1  1];
%R0(:,:,1)=[1  1  1
%          1  1  1
%          1  1  1];
%R0(:,:,2)=[1  1  1
%          1  1  1
%          1  1  1];
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
    Essr1(i,1)=Essr(i,i,1);% pi0(1,1)=pi1(1,1), expectation
    Essr2(i,1)=Essr(i,i,2);% pi0(1,2)
    Essr3(i,1)=Essr(i,i,3);% pi1(2,1)
    Essr4(i,1)=Essr(i,i,4);% pi0(2,2)=pi1(2,2)
end

Pi0=[0,0,0
     0,0,1
     0,1,0
     0,1,1
     1,0,0
     1,0,1
     1,1,0
     1,1,1];
Pi1=[0,0,0
     0,0,1
     0,1,0
     0,1,1
     1,0,0
     1,0,1
     1,1,0
     1,1,1];
gamma=0.95;% discount factor gamma;
alpha=0.5;%equilibrium parameter£¬V0 equal alpha,V1 equal 1-alpha
Max=200;%The largest number of iterations
delta=0.00001;
G=zeros(Ns1*Ns0,64);%Record the V value after each convergence in Agent0
H=zeros(Ns1*Ns0,64);%Record the V value after each convergence in Agent1
GH=zeros(Ns1*Ns0,64);%Record the V value of the entire system after each convergence

%Pi00=ceil(2*rand(1,3))Randomly generate a set of policies
%Pi00=[1,1,1];%Agent0 will execute actions
%Pi11=[2,2,2];%Agent1 will execute actions
%(Pi0,Pi1)=(Pi00(x),Pi11(y))
for i=1:8
   Pi00=Pi0(i,:)+[1,1,1]; 
   for j=1:8%Initialize the strategy function, an Ns column vector, different values of k, corresponding to different strategies
      Pi11=Pi1(j,:)+[1,1,1];
      V0=zeros(Ns0*Ns1,1);%It produces the full zero square matrix of Ns0 times Ns1 dimensions
      V1=zeros(Ns1*Ns0,1);
      V=zeros(Ns1*Ns0,1);
      stop=1;
      while stop>0
          for s=1:Ns0
              for t=1:Ns1
                  Delta0=0;
                  a0=Pi00(s);%%The initial strategy for agent0
                  a1=Pi11(t);%%The initial strategy for agent1
                  v0=V0((s-1)*Ns1+t);%V0 is n rows and 1 columns, and v0 is a value
                  v1=V1((s-1)*Ns1+t);
                  v=V((s-1)*Ns1+t);
                  if(a0==1 && a1==1)
                      V0((s-1)*Ns1+t)=Essr1((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,1)*gamma*V;%Value = probability * (reward value + discount factor * value), V function of agent0
                      V1((s-1)*Ns1+t)=Essr1((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,1)*gamma*V;%Agengt1 execute a1,Agent0 execute a1
                      V((s-1)*Ns1+t)=alpha* V0((s-1)*Ns1+t)+(1-alpha)*V1((s-1)*Ns1+t);
                      Delta0=max(Delta0,abs(V((s-1)*Ns1+t)-v));
                  elseif(a0==1 && a1==2)
                      V0((s-1)*Ns1+t)=Essr2((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,2)*gamma*V;%Agengt1 execute a1,Agent0 execute a2  
                      V1((s-1)*Ns1+t)=Essr3((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,3)*gamma*V;%At this point, for Agent1, the update is still performed as a2 at Agengt1 and a1 at Agent0
                      V((s-1)*Ns1+t)=alpha* V0((s-1)*Ns1+t)+(1-alpha)*V1((s-1)*Ns1+t);
                      Delta0=max(Delta0,abs(V((s-1)*Ns1+t)-v));
                  elseif(a0==2 && a1==1)
                      V1((s-1)*Ns1+t)=Essr3((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,3)*gamma*V;%Agengt1 execute a2,Agent0 execute a1
                      V0((s-1)*Ns1+t)=Essr2((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,2)*gamma*V;%At this point, for Agent0, the update is still performed as a1 at Agengt1 and a2 at Agent0 
                      V((s-1)*Ns1+t)=alpha* V0((s-1)*Ns1+t)+(1-alpha)*V1((s-1)*Ns1+t);
                      Delta0=max(Delta0,abs(V((s-1)*Ns1+t)-v));
                  else
                      V0((s-1)*Ns1+t)=Essr4((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,4)*gamma*V;%Agengt1 execute a2,Agent0 execute a2
                      V1((s-1)*Ns1+t)=Essr4((s-1)*Ns1+t,1)+T00((s-1)*Ns1+t,:,4)*gamma*V;
                      V((s-1)*Ns1+t)=alpha* V0((s-1)*Ns1+t)+(1-alpha)*V1((s-1)*Ns1+t);
                      Delta0=max(Delta0,abs(V((s-1)*Ns1+t)-v));
                  end
              end
          end
          if (Delta0<delta)%When the error is less than, stop the iteration 
              stop=0;
          end
      end
      G(:,(i-1)*8+j)=V0;
      H(:,(i-1)*8+j)=V1;
      GH(:,(i-1)*8+j)=V;
   end
end

%V_final=alpha*G+(1-alpha)*H;
s=1:9;
k=0:63;
plot(k,GH(s,k+1),'-') 
hold on
%plot(k,GH(1,k+1),'.b', k,GH(2,k+1),'*g', k,GH(3,k+1),'or', k,GH(4,k+1),'xc', k,GH(5,k+1),'+m', k,GH(6,k+1),'sy', k,GH(7,k+1),'dk', k,GH(8,k+1),'^b', k,GH(9,k+1),'.pg') 
plot(k,GH(1,k+1), k,GH(2,k+1), k,GH(3,k+1), k,GH(4,k+1), k,GH(5,k+1), k,GH(6,k+1), k,GH(7,k+1), k,GH(8,k+1), k,GH(9,k+1)) 

title('Convergence comparison graph under all strategies');

end%funciton