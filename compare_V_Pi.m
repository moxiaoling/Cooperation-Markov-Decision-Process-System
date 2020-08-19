function [V,v,Delta]= compare_V_Pi()
%Mr. Xu wrote the program,
%The existence of the optimal strategy
clear
Ns=3;   %the number of states
Na=2;   %the number of actions
S=[1,2,3]; %state set
A=[1,2];   %action set
%Probability transfer function T(s,a,s'), rewrite to T(s,s',a)
T(:,:,1)=[0.4  0  0.6
          0.7  0.1  0.2
          0.5  0  0.5];

T(:,:,2)=[0.2  0  0.8
          0  0.9   0.1
          0.3  0.4  0.3];
      
%Reward functionR(s,a,s'). rewrite to R(s,s',a)     
R(:,:,1)=[1  0  1
          4  1  1
          1  0  1];

R(:,:,2)=[1  0  3
          0  1  3
          -1  1  2];
Rmax=max(max(max(R))); %Maximum reward value in the system
%Define the discount factorgamma;  
gamma=0.95;
%Error control parameter
delta=0.0001;
%Max=200; %Maximum number of iteration steps
V=zeros(Ns,1); %initial V-function£¬All zero column vectors of Ns dimension, The initial value of V is modified without affecting the value of final probability convergence
Pi0=[0,0,0
     0,0,1
     0,1,0
     0,1,1
     1,0,0
     1,0,1
     1,1,0
     1,1,1]; %3 states, 2 actions, 2x2x2=8 choices
G=zeros(Ns,8); %Keep track of the rewards you get for each option

 for k=1:8
   Pi=Pi0(k,:)+[1,1,1]; %Initialize the strategy function, an Ns column vector, different values of k, corresponding to different strategies
   stop=1;
   while stop>0
      Delta=0;
      for s=1:Ns %For example, use S =1 to find out V=[1,0,0]'. When s=2, use the value of V=[1,4.465,0]' to find out V=[1,4.465,0]',
                 %keep iterating, and finally find out the value of V under Ns states. After one round of the for loop, 
                 %the value of two adjacent deltas is determined. If the condition is not satisfied, the iteration starts from the for loop again. 
                 %At the end of the iteration, a final stable V value (convergent value) is generated.
        v=V(s);
        a=Pi(s);
        Ta=T(s,:,a); %Ta is the probability shift
        Ra=R(s,:,a); %Ra is the reward 
        V(s)=Ta*(Ra'+gamma*V);%Value = probability * (award value + discount factor * value)
        Delta=max(Delta, abs(v-V(s))); %The difference between the absolute value of the value function and the error
      end
      if Delta<delta %When the error is less than, stop the iteration
        stop=0;  
      end
   end
   G(:,k)=V;
end

s=1:Ns;
k=0:7;
  plot(k,G(s,k+1),'-') 
  hold on
  plot(k,G(1,k+1),'.', k,G(2,k+1),'*', k,G(3,k+1),'.') 
title('Convergence Comparison Graphs Undes All Policy')
end



