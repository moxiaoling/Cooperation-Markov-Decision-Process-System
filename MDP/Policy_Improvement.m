function [Pi,stable]=Policy_Improvement(V,Pi)
%Initial value function V_0, the initial policy can be randomly generated.
%By Policy_Evaluation(V_k,Pi_k) running once, a convergent value function V_{k+1} is obtained under policy Pi_k.
%V_{k+1} is the stable value function of Pi_k.
%Policy_Improvement(V_{k+1},Pi_k)£¬Under the idea of dynamic programming, an optimal strategy for V_{k+1} Pi_{k+1} is obtained.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%define discount factor gamma;  
gamma=0.95;
Ns=3;   %state number
Na=2;   %action number

%%Basic parameters of the system: probability transfer function T(s, A,s'), reward function R(s, A,s')

%%Probability transfer functionT(s,a,s')£¨3d matrix£©, rewrite to T(s,s',a)
T(:,:,1)=[0.5  0  0.5
          0.7  0.1  0.2
          0.4  0  0.6];

T(:,:,2)=[0  0  1
          0  0.95  0.05
          0.3  0.3  0.4];
      
%Reward function R(s,a,s')£¨3d matrix£©, rewrite to R(s,s',a)   
R(:,:,1)=[1  0  1
          5  1  1
          1  1  1];

R(:,:,2)=[0  0  1
          0  1  1
          -1  1  1];
      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stable=1;
for s=1:Ns
    b=Pi(s);
    V_Action=zeros(1,Na);
    for a=1:Na
        %Construct the probability matrix and value matrix under behavior a
        Pa=zeros(Ns);
        Ra=zeros(Ns);
        for j=1:Ns
            Pa(s,:)=T(s,:,a);
            Ra(s,:)=R(s,:,a);
        end
       V_Action(a)=Pa(s,:)*Ra(s,:)'+gamma*Pa(s,:)*V;
    end
    [Maxv,Maxa]=max(V_Action); %Take Maxa, where the maximum Maxv is in V_Action
    Pi(s)=Maxa;
    if ~(b==Maxa) 
        stable=0;
        break;
    end
end
end




