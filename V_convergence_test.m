function [V,C0,Pi0,Total]= V_convergence_test()
%Write by Mo Xiaoling
%Verify that the probability of the corresponding initial V0(S) is different without the Pi strategy,
%But the Vk(S) value of the final convergence is stable

Ns=5;   %The number of states
Na=3;   %The number of actions
%Reward function: R(s,a)£ºThe reward value obtained by executing action A in state S
Rmax=4; %Maximum reward value
%Rsareward value
%R1=rand(Ns,Ns);%reward value
% R1=[0.1690    0.5470    0.1835    0.9294    0.3063
%     0.6491    0.2963    0.3685    0.7757    0.5085
%     0.7317    0.7447    0.6256    0.4868    0.5108
%     0.6477    0.1890    0.7802    0.4359    0.8176
%     0.4509    0.6868    0.0811    0.4468    0.7948];
R1=[1      0    0     0     0
    1/3    0    1/3   0     1/3
    0      0    0     1/3   2/3
    1/2    0    0     1/2   0 
    0    1/4    0     3/4   0];
Rsa=Rmax*R1;%reward value for state to state
r=0.85;  %discount factor

%R=rand(Ns,Ns);%The values are randomly generated and the probability matrix is obtained after normalization
% R=[0.6443    0.9390    0.2077    0.1948    0.3111
%     0.3786    0.8759    0.3012    0.2259    0.9234
%     0.8116    0.5502    0.4709    0.1707    0.4302
%     0.5328    0.6225    0.2305    0.2277    0.1848
%     0.3507    0.5870    0.8443    0.4357    0.9049];
R=[2 0 0 0 0
    2 0 2 0 3
    0 0 0 1 2
    1 0 0 2 0
    0 1 0 2 0];
c=sum(R,2); %The N vector formed by the row sum as a probability normalization factor

%Generate the probability transition matrix P
P=zeros(Ns,Ns); 
for i=1:Ns
    P(i,:)=R(i,:)/c(i);
end %Generate Ns*Ns dimension probability matrix P

%Initializes the state set and the behavior set£º
S=[1,2,3,4,5];
A=[1,2,3];

%Policy function Pi£ºS-->A define:
Pi=ceil(3*rand(1,5)); %Randomly-defined policy
%Pi=ceil(Na*rand(1,Ns)); 
%Pi=[1     2     1     3     1];%If you change the value of Pi, you change the initial probability distribution

%V =[12.7515   13.0732   13.8081   13.1210   13.2857 ]
%Total = 66.0395
%C =[56    56    56    56    56]
%Characteristics of the optimal strategy: for each s, there is a certain n. After every n step, Vn(s) rises monotonically¡£

%Probability distribution Dsa was defined; Assume that: Dsa=P(s*a,:). 
%Take every s*a row in P as the probability distribution corresponding to (s,a)


%Rsa(s,Pi(s)) The reward value obtained under strategy Pi. (Column vector)
%The value of the reward obtained by executing the Pi strategy at each status point
R0=ones(Ns,0); %initialize
for i=1:Ns
   R0(i,1)=Rsa(i,Pi(i));
end

%initialize V^{Pi}
V0=R0;  %R0(i,1)=R(i,Pi(i))

%Iteration calculation of Bellman square iteration£º
M=1;
Max=60; %Maximum number of iteration steps
VPI=zeros(Ns,Max);
VPI(:,1)=V0;
C=ones(Ns,1); %Step convergence
mu=0.001/Rmax; %Convergent error control parameters

Essr=P*(Rsa');%The probability of a state transition to a 
%state * the value of the state to state reward = the expected value of the state to state reward

for M=1:Max-1
   for s=1:Ns
      a=Pi(s);%The value of A has nothing to do with M, only with S
      VPI(s,M+1)=(Essr(s,s)+r*(P(s,:)*VPI(:,M)));%Inconsistent results, by moxiaoling
      if abs(VPI(s,M+1)-VPI(s,M))>=mu
          C(s,1)=C(s,1)+1;
      end
   end
end

%Data convergence diagram
hold off
s=1:Ns;
x=1:Max;
  plot(x,VPI(s,x),'-')
title('V^{Pi}(s) Convergent Graph')


Pi0=Pi
 V=VPI(:,Max)'%The final value of VPI
 Total =sum(VPI(:,Max))
 C0=C'

 hold off

end