%Enter a Markov decision instance
%clear;
%System parameter definition: state set, behavior set, probability transfer function T(s, A,s'), reward function R(s, A,s')
Ns0=3;   %state number
Na0=2;   %action number
S0=[1,2,3]; %state set
A0=[1,2];   %action set
Ns1=3;   %state number
Na1=2;   %action number
S1=[1,2,3]; %state set
A1=[1,2];   %action set
%Probability transfer function T(s, A,s') (3d matrix), rewritten as T(s,s',a)
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
%Rmax=max(max(max(R))); %Maximum reward value
%define discount factor gamma;  
gamma=0.95;
delta=0.00001; %Convergent error control parameters

%Initialize the value function V, and initialize the strategy function Pi

%V=rand(Ns0*Ns1,1);
V=[1;1;1;1;1;1;1;1;1];%Initialize a V function, randomly pick an Ns column vector
Pi=ceil(Na0*rand(1,Ns0+Ns1));%Initializes the policy function Pi, an Ns - dimensional column vector
%Pi=[1,2,1,2,1,2];
%Pi=[2,2,2,2,2,2];
%Pi=[2,1,2,2,2,2];


V=Policy_Evaluation_test(V,Pi);

[Pi,stable]=Policy_Improvement_test(V,Pi);

% Iteration under instability
Step=1;
while stable<1
  V=Policy_Evaluation_test(V,Pi);
  [Pi,stable]=Policy_Improvement_test(V,Pi);
  Step=Step+1;
end
disp('The optimal policy£º');
Opt_Pi_pair=[Pi([1,2,3]);Pi([4,5,6])]
disp('Find out the number of iteration of optimal strategy cycle evolution£º');
StepC=Step
disp('The Value function of')
V



