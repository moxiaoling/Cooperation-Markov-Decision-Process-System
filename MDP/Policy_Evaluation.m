function V=Policy_Evaluation(V,Pi)
%Initial value function V_0, the initial policy can be randomly generated.
%By Policy_Evaluation(V_k,Pi_k) running once, a convergent value function V_{k+1} is obtained under policy Pi_k.
%V_{k+1} is the stable value function of Pi_k.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%define discount factor gamma;  
gamma=0.95;
%Convergence error control parameters during evolution (iteration)
delta=0.00001; 

%Basic parameters of the system: probability transfer function T(s, A,s'), reward function R(s, A,s')
Ns=3;   %state number
%Probability transfer functionT(s,a,s')��3d matrix��, rewrite to T(s,s',a)
T(:,:,1)=[0.5  0  0.5
          0.7  0.1  0.2
          0.4  0  0.6];

T(:,:,2)=[0  0  1
          0  0.95  0.05
          0.3  0.3  0.4];
      
%Reward function R(s,a,s')��3d matrix��, rewrite to R(s,s',a)     
R(:,:,1)=[1  0  1
          5  1  1
          1  1  1];

R(:,:,2)=[0  0  1
          0  1  1
          -1  1  1];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stop=1;
while stop>0
    Delta=0;
    for s=1:Ns
        v=V(s);
        Ta=T(s,:,Pi(s));  %The probability matrix Ta of Markov system and the reward function 
                               %Ra are generated by T- function and R- function of the system and according to the strategy Pi
        Ra=R(s,:,Pi(s));
        V(s)=Ta*(Ra'+gamma*V);
        Delta=max(Delta, abs(v-V(s)));
    end
    if Delta<delta  
        stop=0;  
    end
end
%V; %Iterate over the output value function
end

