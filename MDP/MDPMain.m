%Enter a Markov decision instance
clear;
%System parameter definition: state set, behavior set, probability transfer function T(s, A,s'), reward function R(s, A,s')
S=[1,2,3]; %state set
A=[1,2];   %action set
Ns=3;   %state number
Na=2;   %action number
%Probability transfer function T(s, A, S ') (3d matrix), rewrite to T(s,s',a)
T(:,:,1)=[0.5  0  0.5
          0.7  0.1  0.2
          0.4  0  0.6];

T(:,:,2)=[0  0  1
          0  0.95  0.05
          0.3  0.3  0.4];
      
%Reward function R(s,a,s')£¨3d matrix£©. rewrite to R(s,s',a)     
R(:,:,1)=[1  0  1
          5  1  1
          1  1  1];

R(:,:,2)=[0  0  1
          0  1  1
          -1  1  1];
      
Rmax=max(max(max(R))); %Maximum reward value
%define discount factor gamma;  
gamma=0.95;
delta=0.00001; %Convergent error control parameters

%Initialize the value function V, and initialize the strategy function Pi

V=Rmax*rand(Ns,1); %Initialize a V function, randomly pick an Ns column vector
Pi=ceil(Na*rand(Ns,1));%Initializes the policy function Pi, an Ns - dimensional column vector

%In this example £ºPi=[2,1,2] is the optimal policy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Evolution: the current value function V_k and the current strategy Pi_k evolve to generate the value function V_{k+1}.
% The value function V_{k+1} is stable under the policy Pi_k. (Because of convergence)
V=Policy_Evaluation(V,Pi);
%Based on the value function V_{k+1}, the optimal strategy of Pi_{k+1} under V_{k+1} is obtained by using dynamic 
%programming algorithm. And the stability of Pi_{k+1} about V_{k+1} is verified.
%Stability verification: for each state S,Pi_{k+1}(s)=Pi_k(s). If some s is found, Pi_{k+1}(s)=Pi_k(s) is not true, then it is unstable.
[Pi,stable]=Policy_Improvement(V,Pi);

%If there has a s: Pi_{k+1}(s) not equal to Pi_k(s), Pi_{k+1} unstable¡£
%If Pi_{k+1}stable£¬we have find the optimal policy£¬otherwise, running Policy_Evaluation(V_{k+1},Pi_{k+1})(Enter a new round of evolution)

% Iteration under instability
Step=1;
while stable<1
  V=Policy_Evaluation(V,Pi);
  [Pi,stable]=Policy_Improvement(V,Pi);
  Step=Step+1;
end
disp('The optimal policy£º');
Opt_Pi=Pi'
disp('Find out the number of iteration of optimal strategy cycle evolution£º');
StepC=Step



