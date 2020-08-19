function V=Policy_Evaluation_test(V,Pi)
%Initial value function V_0, the initial policy can be randomly generated.
%By Policy_Evaluation(V_k,Pi_k) running once, a convergent value function V_{k+1} is obtained under policy Pi_k.
%V_{k+1} is a stable value function for Pi_k.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Convergence error control parameters during evolution (iteration)
delta=0.00001; 

%Basic parameters of the system: probability transfer functionT(s,a,s’），reward functionR(s,a,s')
Ns0=3;   %state number
Na0=2;   %action number
S0=[1,2,3]; %state set
A0=[1,2];   %action set
Ns1=3;   %state number
Na1=2;   %action number
S1=[1,2,3]; %state set
A1=[1,2];   %action set
%probability transfer functionT(s,a,s')（3d matrix）, rewrite to T(s,s',a)
T00(:,:,1)=[0.2  0.3  0.5
            0.7  0.1  0.2
            0.3  0.1  0.6];
T00(:,:,2)=[0.4  0.5  0.1
            0.1  0.4  0.5
            0.3  0.3  0.4];
T01(:,:,1)=[0   0    1
            0   0.95  0.05
            0.3  0.3  0.4];
T01(:,:,2)=[0.3   0.4    0.3
            0.5   0.1    0.4
            0.3   0    0.7];
%Reward function R(s,a,s'). rewrite to R(s,s',a)     
R00(:,:,1)=[1  1  1
            2  1  1
            1  1  2];
R00(:,:,2)=[1  1  1
            1  4  1
           -1  1  1];

R01(:,:,1)=[0  0  1
            0  1  2
            3  1  1];
R01(:,:,2)=[1  1  3
           -1  5  -1
            1  0  1];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%T represents the tensor product of the probability transfer of two agent states, the probability of the transfer of Agent1
% after the transfer of Agent0, and T00 represents the tensor product of the behavior
T(:,:,1)=kron(T00(:,:,1),T01(:,:,1));%Agent0 excute a0,Agent1 excute a0
T(:,:,2)=kron(T00(:,:,1),T01(:,:,2));%Agent0 excute a0,Agent1 excute a1
T(:,:,3)=kron(T00(:,:,2),T01(:,:,1));%Agent0 excute a1,Agent1 excute a0
T(:,:,4)=kron(T00(:,:,2),T01(:,:,2));%Agent0 excute a1,Agent1 excute a1
%%R represents the tensor product of the probability transfer of two agent states; after the transfer of Agent0, 
%the reward value of the transfer of Agent1; R01 represents the reward value of the behavior
R(:,:,1)=kron(R00(:,:,1),R01(:,:,1));
R(:,:,2)=kron(R00(:,:,1),R01(:,:,2));
R(:,:,3)=kron(R00(:,:,2),R01(:,:,1));
R(:,:,4)=kron(R00(:,:,2),R01(:,:,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%T1 represents the tensor product of the probability transfer of two agent states, after the transfer of Agent0, 
%the probability of the transfer of Agent1, and T00 represents the tensor product of the behavior
T1(:,:,1)=kron(T01(:,:,1),T00(:,:,1));%Agent0执行a0,Agent1执行a0
T1(:,:,2)=kron(T01(:,:,1),T00(:,:,2));%Agent0执行a0,Agent1执行a1
T1(:,:,3)=kron(T01(:,:,2),T00(:,:,1));%Agent0执行a1,Agent1执行a0
T1(:,:,4)=kron(T01(:,:,2),T00(:,:,2));%Agent0执行a1,Agent1执行a1
%%R1 represents the tensor product of the probability transfer between two agents,
%Agent1 represents the reward value of Agent0 after the transfer, and R01 represents the reward value of the behavior
R1(:,:,1)=kron(R01(:,:,1),R00(:,:,1));
R1(:,:,2)=kron(R01(:,:,1),R00(:,:,2));
R1(:,:,3)=kron(R01(:,:,2),R00(:,:,1));
R1(:,:,4)=kron(R01(:,:,2),R00(:,:,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:4
    Essr0(:,:,i)=(T(:,:,i)*R(:,:,i)');%%The value expectations of each state under different strategies
    Essr1(:,:,i)=(T1(:,:,i)*R1(:,:,i)');%%The value expectations of each state under different strategies
end
for i=1:9
    Essr01(i,1)=Essr0(i,i,1);% pi0(1,1)=pi1(1,1), expectations
    Essr02(i,1)=Essr0(i,i,2);% tpi0(1,2)
    Essr03(i,1)=Essr0(i,i,3);% pi1(2,1)
    Essr04(i,1)=Essr0(i,i,4);% pi0(2,2)=pi1(2,2)
    Essr11(i,1)=Essr1(i,i,1);% pi0(1,1)=pi1(1,1), expectations
    Essr12(i,1)=Essr1(i,i,2);% tpi0(1,2)
    Essr13(i,1)=Essr1(i,i,3);% pi1(2,1)
    Essr14(i,1)=Essr1(i,i,4);% pi0(2,2)=pi1(2,2)
end
  
gamma=0.95;%define discount factor gamma;
alpha=0.5;%Define the equilibrium parameter，V0 equal to alpha,V1 equal to 1-alpha
%Max=500;%The largest number of iterations
V0=ones(Ns0*Ns1,1);%Output the Ns0*Ns1 dimension full zero square matrix and record the value function
V1=ones(Ns1*Ns0,1);

stop=1;
      while stop>0
          Delta0=zeros(Ns0*Ns1,1);
          for s=1:Ns0
              for t=1:Ns1
                  %Delta0=zeros(Ns0*Ns1,1);
                  %Delta0=0
                  a0=Pi(s);%%The initial strategy for agent0
                  a1=Pi(Ns0+t);%%The initial strategy for agent1
                  v=V((s-1)*Ns1+t);
                  if(a0==1 && a1==1)
                      V0((s-1)*Ns1+t)=Essr01((s-1)*Ns1+t,1)+T((s-1)*Ns1+t,:,1)*gamma*V;%Value = probability * (reward value + discount factor * value), V function of agent 0 
                      V1((s-1)*Ns1+t)=Essr11((s-1)*Ns1+t,1)+T1((s-1)*Ns1+t,:,1)*gamma*V;%Agengt1 excute a1,Agent0 excute a1
                      V((s-1)*Ns1+t)=alpha* V0((s-1)*Ns1+t)+(1-alpha)*V1((s-1)*Ns1+t);
                      %Delta0=max(Delta0,abs(V((s-1)*Ns1+t)-v));
                      Delta0((s-1)*Ns1+t)=max(Delta0((s-1)*Ns1+t),abs(V((s-1)*Ns1+t)-v));
                  elseif(a0==1 && a1==2)
                      V0((s-1)*Ns1+t)=Essr02((s-1)*Ns1+t,1)+T((s-1)*Ns1+t,:,2)*gamma*V;%Agengt1 excute a1,Agent0 excute a2  
                      V1((s-1)*Ns1+t)=Essr13((s-1)*Ns1+t,1)+T1((s-1)*Ns1+t,:,3)*gamma*V;%At this point, for Agent1, the update is still performed as a2 at Agengt1 and a1 at Agent0
                      V((s-1)*Ns1+t)=alpha* V0((s-1)*Ns1+t)+(1-alpha)*V1((s-1)*Ns1+t);
                      %Delta0=max(Delta0,abs(V((s-1)*Ns1+t)-v));
                      Delta0((s-1)*Ns1+t)=max(Delta0((s-1)*Ns1+t),abs(V((s-1)*Ns1+t)-v));
                  elseif(a0==2 && a1==1)
                      V1((s-1)*Ns1+t)=Essr03((s-1)*Ns1+t,1)+T((s-1)*Ns1+t,:,3)*gamma*V;%Agengt1 excute a2,Agent0 excute a1
                      V0((s-1)*Ns1+t)=Essr12((s-1)*Ns1+t,1)+T1((s-1)*Ns1+t,:,2)*gamma*V;%At this point, for Agent0, we're still going to do a1 with Agengt1, and We're going to do a2 with Agent0
                      V((s-1)*Ns1+t)=alpha* V0((s-1)*Ns1+t)+(1-alpha)*V1((s-1)*Ns1+t);
                      %Delta0=max(Delta0,abs(V((s-1)*Ns1+t)-v));
                      Delta0((s-1)*Ns1+t)=max(Delta0((s-1)*Ns1+t),abs(V((s-1)*Ns1+t)-v));
                  else
                      V0((s-1)*Ns1+t)=Essr04((s-1)*Ns1+t,1)+T((s-1)*Ns1+t,:,4)*gamma*V;%Agengt1 excute a2,Agent0 excute a2
                      V1((s-1)*Ns1+t)=Essr14((s-1)*Ns1+t,1)+T1((s-1)*Ns1+t,:,4)*gamma*V;
                      V((s-1)*Ns1+t)=alpha* V0((s-1)*Ns1+t)+(1-alpha)*V1((s-1)*Ns1+t);
                      %Delta0=max(Delta0,abs(V((s-1)*Ns1+t)-v));
                      Delta0((s-1)*Ns1+t)=max(Delta0((s-1)*Ns1+t),abs(V((s-1)*Ns1+t)-v));
                  end
              end
          end
          if (max(Delta0)<delta)%When the error is less than, stop the iteration
              stop=0;
          end
      end
V;%Iterate over the output value function
end


