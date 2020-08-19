function [V,C0,Pi0,Total]= V_convergence()
%experience findings£º
%1. For a given Rsa, under the optimal strategy Pi, there is an N step for every S, and after n, Vn(s) rises monotonically.
%2. For a reward function with a form such as Rsa=Rmax*R,V^{Pi} is independent of the scaling coefficient Rmax for a fixed strategy Pi.
%3. There is a critical value Rmax0, when Rmax<=Rmax, Pi is the optimal strategy.



Ns=5;   %state number
Na=3;   %action number
%Reward function: R value function R(s,a) : the reward value obtained by performing an action A in state S
Rmax=4; %Maximum reward value
%Rsa  reward value
Rsa=Rmax*[0.3993    0.2920    0.1062
          0.5269    0.4317    0.3724
          0.4168    0.0155    0.1981
          0.6569    0.9841    0.4897
          0.6280    0.1672    0.3395];
    
%Define discount factor: The discount factor of the R(s,a) value is taken down after each step
  r=0.85;  
%The results show that the convergence value and convergence speed have a great relationship with r value.
% The greater the r value, the greater the convergence V value and the slower the convergence speed.

%Each pair (s,a) specifies a probability distribution, Dsa[X=s']: the probability of turning to state S 'after performing an action A under state S.

N=Ns*Na; %Total probability distribution number:
%Generation of probability matrix P:
%Randomly generate a nonnegative matrix
%R=rand(N,Ns); %N*NsNonnegative matrix, using this to define all transition probabilities
R=[0.5309    0.2810    0.2548    0.7702    0.2691
    0.6544    0.4401    0.2240    0.3225    0.7655
    0.4076    0.5271    0.6678    0.7847    0.1887
    0.8200    0.4574    0.8444    0.4714    0.2875
    0.7184    0.8754    0.3445    0.0358    0.0911
    0.9686    0.5181    0.7805    0.1759    0.5762
    0.5313    0.9436    0.6753    0.7218    0.6834
    0.3251    0.6377    0.0067    0.4735    0.5466
    0.1056    0.9577    0.6022    0.1527    0.4257
    0.6110    0.2407    0.3868    0.3411    0.6444
    0.7788    0.6761    0.9160    0.6074    0.6476
    0.4235    0.2891    0.0012    0.1917    0.6790
    0.0908    0.6718    0.4624    0.7384    0.6358
    0.2665    0.6951    0.4243    0.2428    0.9452
    0.1537    0.0680    0.4609    0.9174    0.2089];

c=sum(R,2); %The N vector formed by the row sum as a probability normalization factor

%Generate the probability transition matrix P
P=zeros(N,Ns); 
for i=1:N
    P(i,:)=R(i,:)/c(i);
end %It produces Ns by Ns probability matrix P

%Initialize state sets and behavior sets:
S=[1,2,3,4,5];
A=[1,2,3];


Pi=[1     2     1     3     1];

R0=ones(Ns,0); %Initialize
for i=1:Ns
   R0(i,1)=Rsa(i,Pi(i));
end

V0=R0;  %R0(i,1)=R(i,Pi(i))

%Iteration calculation of Bellman square iteration£º
M=1;
Max=200; %Maximum number of iteration steps
VPI=zeros(Ns,Max);
VPI(:,1)=V0;
C=ones(Ns,1); %Step convergence
mu=0.001/Rmax; %Convergent error control parameters



%%%%%%%%%%%%%%%%%%%by_moxiaoling
%Pss is the probability of going from state S to state S, the square matrix of Ns*Ns
for i=1:Ns
Pss(i,:)=sum(P(Na*i-Na+1:Na*i,:));
end
Pss=Pss/Na;%The probability matrix is normalized

%Rss represents the reward value obtained from state S to state S, the square matrix of Ns*Ns
for i=1:Ns
for j=1:Ns
Rss(i,j)=Rsa(i,:)*P((Na*i-Na+1:Na*i),j);
end
end
Essr=Pss*(Rss');%The probability of a state transition to a state * the value of the state to state reward = the expected value of the state to state reward
Pss1=Pss';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%by_moxiaoling

for M=1:Max-1
   for s=1:Ns
      a=Pi(s);%The value of A has nothing to do with M, only with S
   %   VPI(s,M+1)=R(s,a)+r*(P(s*a,:)*VPI(:,M));
      VPI(s,M+1)=(Essr(s,s)+r*(Pss(s,:)*VPI(:,M)));
     
       
      if abs(VPI(s,M+1)-VPI(s,M))>=mu
          C(s,1)=C(s,1)+1;
      end
   end
end


hold off
s=1:Ns;
x=1:Max;
  plot(x,VPI(s,x),'-.')
title('V^{Pi}(s)convergent graph')


Pi0=Pi
 V=VPI(:,Max)'%The final value of VPI
 Total =sum(VPI(:,Max))%The sum of the final VPI values after the iteration
 C0=C'

 hold off
% 
% for s=1:Ns
%  subplot(Ns,2,s);
%   x=1:Max;
%   plot(x,VPI(s,x),'--')
% end
end