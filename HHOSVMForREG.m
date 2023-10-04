%
% Harris hawks optimization Algorithm - SVR
%
function [cost,c,g,e,bestFitness] = HHOSVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, population, max_iteration)

%Parameters
pop = population;
max_iter = max_iteration;
part_dim = 3;
Ub = [Crange(2), gamma(2), epsilon(2)];
Lb = [Crange(1), gamma(1), epsilon(1)];

% initialize the location and Energy of the rabbit
Rabbit_Location=zeros(1,part_dim);
Rabbit_Energy=inf;

%preallocation of butterfly position and fitness
X = zeros(pop,part_dim); %pre_allocation of X butterfly position
fitness = zeros(pop,1); %pre_allocation of global fitness function (MSE) value

CNVG=zeros(1,max_iter);

%%%%% Initialize position and evaluate initial fitness

for i=1:pop
    X(i,1) = Crange(1)+(Crange(2)-Crange(1)) * rand(1);
    X(i,2) = gamma(1)+(gamma(2)-gamma(1)) * rand(1);
    X(i,3) = epsilon(1)+(epsilon(2)-Crange(1)) * rand(1);
   
end
t= 0;
while t < max_iter %its ietration number
     Iteration = t
     
    for i=1:pop
        
        % Check boundries
        X(i,:) = max(X(i,:), Lb);
        X(i,:) = min(X(i,:), Ub);
        
        % fitness of locations
        X(i,1) = Crange(1)+(Crange(2)-Crange(1)) * rand(1);
        X(i,2) = gamma(1)+(gamma(2)-gamma(1)) * rand(1);
        X(i,3) = epsilon(1)+(epsilon(2)-Crange(1)) * rand(1);
    
        C = X(i,1);
        gam = X(i,2);
        epsil = X(i,3);
    
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);    
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
    
        fitness(i) = accuracy(2);
        % Update the location of Rabbit
        if fitness(i)<Rabbit_Energy
            Rabbit_Energy=fitness(i);
            Rabbit_Location=X(i,:);
        end
    end
    
    E1=2*(1-(t/max_iter)); % factor to show the decreaing energy of rabbit
    % Update the location of Harris' hawks
    
    for i=1:pop
        E0=2*rand()-1; %-1<E0<1
        Escaping_Energy=E1*(E0);  % escaping energy of rabbit
        
        if abs(Escaping_Energy)>=1
            % Exploration:
            % Harris' hawks perch randomly based on 2 strategy:
            
            q=rand();
            rand_Hawk_index = floor(pop*rand()+1);
            X_rand = X(rand_Hawk_index, :);
            if q<0.5
                % perch based on other family members
                X(i,:)=X_rand-rand()*abs(X_rand-2*rand()*X(i,:));
            elseif q>=0.5 
                % perch on a random tall tree (random site inside group's home range)
                X(i,:)=(Rabbit_Location-mean(X))-rand()*((Ub-Lb)*rand+Lb);
            end
            
        elseif abs(Escaping_Energy)<1
            % Exploitation:
            % Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
            
            % phase 1: surprise pounce (seven kills)
            % surprise pounce (seven kills): multiple, short rapid dives by different hawks
            
            r=rand(); % probablity of each event
            
            if r>=0.5 && abs(Escaping_Energy)<0.5 % Hard besiege
                X(i,:)=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-X(i,:));
            end
            
            if r>=0.5 && abs(Escaping_Energy)>=0.5  % Soft besiege
                Jump_strength=2*(1-rand()); % random jump strength of the rabbit
                X(i,:)=(Rabbit_Location-X(i,:))-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:));
            end
            
            % phase 2: performing team rapid dives (leapfrog movements)
            if r<0.5 && abs(Escaping_Energy)>=0.5, % Soft besiege % rabbit try to escape by many zigzag deceptive motions
                
                Jump_strength=2*(1-rand());
                X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:));
                
                % Check boundries
                X1 = max(X1, Lb);
                X1 = min(X1, Ub);
                
                C = X1(1);
                gam = X1(2);
                epsil = X1(3);
    
                svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
                model = svmtrain(Ytrain,Xtrain,svmoptions);    
                [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
    
                X1fitness = accuracy(2);
        
        
                if X1fitness < fitness(i) % improved move?
                    X(i,:)=X1;
                else % hawks perform levy-based short rapid dives around the rabbit
                    X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:))+rand(1,part_dim).*Levy(part_dim);
                    
                    % Check boundries
                    X2= max(X2, Lb);
                    X2= min(X2, Ub);
                    
                    C = X2(1);
                    gam = X2(2);
                    epsil = X2(3);
    
                    svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
                    model = svmtrain(Ytrain,Xtrain,svmoptions);    
                    [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
    
                    X2fitness = accuracy(2);
                    
                    if (X2fitness<fitness(i)), % improved move?
                        X(i,:)=X2;
                    end
                end
            end
            
            if r<0.5 && abs(Escaping_Energy)<0.5, % Hard besiege % rabbit try to escape by many zigzag deceptive motions
                % hawks try to decrease their average location with the rabbit
                Jump_strength=2*(1-rand());
                X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X));
                
                % Check boundries
                X1 = max(X1, Lb);
                X1 = min(X1, Ub);                
                
                C = X1(1);
                gam = X1(2);
                epsil = X1(3);
    
                svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
                model = svmtrain(Ytrain,Xtrain,svmoptions);    
                [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
    
                X1fitness = accuracy(2);
        
                if (X1fitness<fitness(i)) % improved move?
                    X(i,:)=X1;
                else % Perform levy-based short rapid dives around the rabbit
                    X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X))+rand(1,part_dim).*Levy(part_dim);
                
                    % Check boundries
                    X2 = max(X2, Lb);
                    X2 = min(X2, Ub);
                    
                    C = X2(1);
                    gam = X2(2);
                    epsil = X2(3);
    
                    svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
                    model = svmtrain(Ytrain,Xtrain,svmoptions);    
                    [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
    
                    X2fitness = accuracy(2);
                    
                    if (X2fitness<fitness(i)), % improved move?
                        X(i,:)=X2;
                    end
                end
            end
            %
        end
    end

    t= t+1;
    CNVG(t)=Rabbit_Energy;

end
   
c = Rabbit_Location(1);
g = Rabbit_Location(2);
e = Rabbit_Location(3);
bestFitness = Rabbit_Energy;
cost = model;
save

end
     
function o=Levy(d)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
o=step;
end
