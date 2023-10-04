%
% Multi-Verse Optimizer - SVR
%
function [cost,c,g,e,bestFitness] = MVOSVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];
%Minimum and maximum of Wormhole Existence Probability
WEP_Max=1;
WEP_Min=0.2;

%preallocation of butterfly position and fitness
X = zeros(pop,part_dim); %pre_allocation of X butterfly position

%%%%% Initialize position and evaluate initial fitness

for i=1:pop
    X(i,1) = Crange(1)+(Crange(2)-Crange(1)) * rand(1);
    X(i,2) = gamma(1)+(gamma(2)-gamma(1)) * rand(1);
    X(i,3) = epsilon(1)+(epsilon(2)-Crange(1)) * rand(1); 
end

%Two variables for saving the position and inflation rate (fitness) of the best universe
Best_universe=zeros(1,part_dim);
Best_universe_Inflation_rate=inf;

for t = 1:max_iter %its ietration number
     Iteration = t
     
    %Eq. (3.3) in the paper
    WEP=WEP_Min+Iteration*((WEP_Max-WEP_Min)/max_iter);
    
    %Travelling Distance Rate (Formula): Eq. (3.4) in the paper
    TDR=1-((Iteration)^(1/6)/(max_iter)^(1/6));
    
    %Inflation rates (I) (fitness values)
    Inflation_rates=zeros(1,pop);
    
    for i=1:pop
        
        % Check if the simple limits/bounds are OK
        X(i,:)= max(X(i,:), Lb);
        X(i,:)= min(X(i,:),Ub);
        % Evaluate new solutions
        C = X(i,1);
        gam = X(i, 2);
        epsil = X(i, 3);
                
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);
    
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
        Inflation_rates(1,i) = accuracy(2);
        
        %Elitism
        if Inflation_rates(1,i)<Best_universe_Inflation_rate
            Best_universe_Inflation_rate=Inflation_rates(1,i);
            Best_universe=X(i,:);
        end      
    end
    
    [sorted_Inflation_rates,sorted_indexes]=sort(Inflation_rates);
    
    for newindex=1:pop
        Sorted_universes(newindex,:)=X(sorted_indexes(newindex),:);
    end
    
    %Normaized inflation rates (NI in Eq. (3.1) in the paper)
    normalized_sorted_Inflation_rates=normr(sorted_Inflation_rates);
    
    X(1,:)= Sorted_universes(1,:);
         
    %Update the Position of universes
    for i=2:pop %Starting from 2 since the firt one is the elite
        Back_hole_index=i;
        for j=1:part_dim
            r1=rand();
            if r1 < normalized_sorted_Inflation_rates(i)
                % for maximization problem -sorted_Inflation_rates should be written as sorted_Inflation_rates
                accumulation = cumsum(-sorted_Inflation_rates);
                p = rand() * accumulation(end);
                chosen_index = -1;
                for index = 1 : length(accumulation)
                    if (accumulation(index) > p)
                        chosen_index = index;
                        break;
                    end
                end
                White_hole_index = chosen_index;
                
                if White_hole_index==-1
                    White_hole_index=1;
                end
                %Eq. (3.1) in the paper
                X(Back_hole_index,j)=Sorted_universes(White_hole_index,j);
            end
                     
            %Eq. (3.2) in the paper if the upper and lower bounds are
            %different for each variables
            r2=rand();
            if r2<WEP
                r3=rand();
                if r3<0.5
                    X(i,j)=Best_universe(1,j)+TDR*((Ub(j)-Lb(j))*rand+Lb(j));
                elseif r3>0.5
                    X(i,j)=Best_universe(1,j)-TDR*((Ub(j)-Lb(j))*rand+Lb(j));
                end
            end
        end
    end
     
end

c = Best_universe(1);
g = Best_universe(2);
e = Best_universe(3);
bestFitness = Best_universe_Inflation_rate;
cost = model;
save 
    
end
     
     
