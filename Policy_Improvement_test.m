function [Pi,stable]=Policy_Improvement_test(V,Pi)
%Initial value function V_0, the initial policy can be randomly generated
%By Policy_Evaluation(V_k,Pi_k) running once, a convergent value function V_{k+1} is obtained under policy Pi_k.
%V_{k+1} is the stable value function of Pi_k.
%Policy_Improvement(V_{k+1},Pi_k)£¬Under the idea of dynamic programming, an optimal strategy Pi_{k+1} about V_{k+1} is obtained.¡£
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%define discount factor gamma;  
gamma=0.95;
Ns0=3;   %state number
Na0=2;   %action number
S0=[1,2,3]; %state set
A0=[1,2];   %action set
Ns1=3;   %state number
Na1=2;   %action number
S1=[1,2,3]; %state set
A1=[1,2];   %action set
%Basic parameters of the system: probability transfer function T(s,a,s¡¯£©£¬Reward function R(s,a,s')
%probability transfer function T(s,a,s')£¨3d matrix£©, rewrite to T(s,s',a)
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
%reward function R(s,a,s'). rewrite to R(s,s',a)     
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%T represents the tensor product of the probability transfer of two agent states, the probability of the transfer of Agent1 
%after the transfer of Agent0, and T00 represents the tensor product of the behavior
T(:,:,1)=kron(T00(:,:,1),T01(:,:,1));%Agent0 excute a0,Agent1 excute a0
T(:,:,2)=kron(T00(:,:,1),T01(:,:,2));%Agent0 excute a0,Agent1 excute a1
T(:,:,3)=kron(T00(:,:,2),T01(:,:,1));%Agent0 excute a1,Agent1 excute a0
T(:,:,4)=kron(T00(:,:,2),T01(:,:,2));%Agent0 excute a1,Agent1 excute a1
%%R represents the tensor product of the probability transfer of two agent states; after the transfer of Agent0,
% the reward value of the transfer of Agent1; R01 represents the reward value of the behavior
R(:,:,1)=kron(R00(:,:,1),R01(:,:,1));
R(:,:,2)=kron(R00(:,:,1),R01(:,:,2));
R(:,:,3)=kron(R00(:,:,2),R01(:,:,1));
R(:,:,4)=kron(R00(:,:,2),R01(:,:,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%T represents the tensor product of the probability transfer of two agent states, the probability of the transfer of Agent1 
%after the transfer of Agent0, and T00 represents the tensor product of the behavior
T1(:,:,1)=kron(T01(:,:,1),T00(:,:,1));%Agent0 excute a0,Agent1 excute a0
T1(:,:,2)=kron(T01(:,:,1),T00(:,:,2));%Agent0 excute a0,Agent1 excute a1
T1(:,:,3)=kron(T01(:,:,2),T00(:,:,1));%Agent0 excute a1,Agent1 excute a0
T1(:,:,4)=kron(T01(:,:,2),T00(:,:,2));%Agent0 excute a1,Agent1 excute a1
%%R represents the tensor product of the probability transfer of two agent states; after the transfer of Agent0, 
%the reward value of the transfer of Agent1; R01 represents the reward value of the behavior
R1(:,:,1)=kron(R01(:,:,1),R00(:,:,1));
R1(:,:,2)=kron(R01(:,:,1),R00(:,:,2));
R1(:,:,3)=kron(R01(:,:,2),R00(:,:,1));
R1(:,:,4)=kron(R01(:,:,2),R00(:,:,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha=0.5;%Define the equilibrium parameter£¬V0 equal to alpha,V1 equal to 1-alpha
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
        
P=Pi;
stable=1;
for s=1:3
        V_Action_0=zeros(Ns0*Ns1,Na0+Na1);%The behavior function from Agent0 to Agent1
        V_Action_1=zeros(Ns0*Ns1,Na0+Na1);%The behavior function from Agent1 to Agent0
        V_Action=zeros(Ns0*Ns1,Na0+Na1);
%for s=1:3
     for t=1:3
        for a0=1:2
            for a1=1:2
                    if (a0==1 && a1==1)
                        V_Action_0((s-1)*Ns1+t,1)=Essr01((s-1)*Ns1+t,1)+gamma*T((s-1)*Ns1+t,:,1)*V;
                        V_Action_1((s-1)*Ns1+t,1)=Essr11((s-1)*Ns1+t,1)+gamma*T1((s-1)*Ns1+t,:,1)*V;
                        V_Action((s-1)*Ns1+t,1)=alpha*V_Action_0((s-1)*Ns1+t,1)+(1-alpha)*V_Action_1((s-1)*Ns1+t,1);
                    elseif(a0==1 && a1==2)
                        V_Action_0((s-1)*Ns1+t,2)=Essr02((s-1)*Ns1+t,1)+gamma*T((s-1)*Ns1+t,:,2)*V;
                        V_Action_1((s-1)*Ns1+t,2)=Essr13((s-1)*Ns1+t,1)+gamma*T1((s-1)*Ns1+t,:,3)*V;
                        V_Action((s-1)*Ns1+t,2)=alpha*V_Action_0((s-1)*Ns1+t,2)+(1-alpha)*V_Action_1((s-1)*Ns1+t,2);
                    elseif(a0==2 && a1==1)
                        V_Action_0((s-1)*Ns1+t,3)=Essr03((s-1)*Ns1+t,1)+gamma*T((s-1)*Ns1+t,:,3)*V;
                        V_Action_1((s-1)*Ns1+t,3)=Essr12((s-1)*Ns1+t,1)+gamma*T1((s-1)*Ns1+t,:,2)*V;
                        V_Action((s-1)*Ns1+t,3)=alpha*V_Action_0((s-1)*Ns1+t,3)+(1-alpha)*V_Action_1((s-1)*Ns1+t,3);
                    else
                        V_Action_0((s-1)*Ns1+t,4)=Essr04((s-1)*Ns1+t,1)+gamma*T((s-1)*Ns1+t,:,4)*V;
                        V_Action_1((s-1)*Ns1+t,4)=Essr14((s-1)*Ns1+t,1)+gamma*T1((s-1)*Ns1+t,:,4)*V;
                        V_Action((s-1)*Ns1+t,4)=alpha*V_Action_0((s-1)*Ns1+t,4)+(1-alpha)*V_Action_1((s-1)*Ns1+t,4);
                    end
            end
        end
     end
     [a,b1]=max(sum(V_Action_0));
     if (b1==1 || b1==2)
         Pi(s)=1;
     else
         Pi(s)=2;
     end
     [a,b2]=max(sum(V_Action_1));
     if (b2==1 || b2==2)
         Pi(s+Ns0)=1;
     else
         Pi(s+Ns0)=2;
     end
end
        if (P == Pi)
            stable=1;
        else
            stable=0;
        end
Pi
end






