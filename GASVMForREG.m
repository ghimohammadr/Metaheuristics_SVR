%
% Genetic Algorithm - SVR
%
function [cost,c,g,e,bestFitness] = GASVMForREG(Ytrain, Xtrain,Ytest,Xtest, Crange, gamma, epsilon, populations, max_iteration)

pop = populations;
max_iter = max_iteration;
part_dim = 3; %default value of dimension
parent_number = pop/2; 
mutation_rate = 0.1; 
minimal_cost = 0.1*10^-6;

cumulative_probabilities = cumsum((parent_number:-1:1) / sum(parent_number:-1:1));
best_fitness = ones(max_iter, 1);
elite = zeros(max_iter, 3);
child_number = pop - parent_number;

%preallocation of position and fitness
X = zeros(pop,part_dim); %pre_allocation of X position
fitness = zeros(pop,1); %pre_allocation of global fitness function (MSE) value

for i=1:pop
    X(i,1) = Crange(1)+(Crange(2)-Crange(1)) * rand(1);
    X(i,2) = gamma(1)+(gamma(2)-gamma(1)) * rand(1);
    X(i,3) = epsilon(1)+(epsilon(2)-Crange(1)) * rand(1);
end

for generation = 1 : max_iter
    Iteration = generation
    for i=1:pop
        C = X(i,1);
        gam = X(i,2);
        epsil = X(i,3);
    
    
        svmoptions = ['-s 3 -t 2 -c ', num2str(C),' -g ',num2str(gam),' -p ',num2str(epsil)];
        model = svmtrain(Ytrain,Xtrain,svmoptions);   
        [predict_label, accuracy, prob_estimates]  = svmpredict( Ytest, Xtest,model);
    
        fitness(i,1) = accuracy(2);
    end
    
    [Gbest_fit, gbestfitindex] = min(fitness); %global fitness and its index value
    Gbest_position = X(gbestfitindex,:); %globalbest position
    
    cost = fitness;
    [cost, index] = sort(cost);
    X = X(index(1:parent_number), :);
    best_fitness(generation) = cost(1);
    elite(generation, :) = X(1, :);

    if best_fitness(generation) < minimal_cost; break; end
    for child = 1:2:child_number % crossover
        mother = min(find(cumulative_probabilities > rand));
        father = min(find(cumulative_probabilities > rand));
        crossover_point = ceil(rand*3);
        mask1 = [ones(1, crossover_point), zeros(1, 3 - crossover_point)];
        mask2 = not(mask1);
        mother_1 = mask1 .* X(mother, :);
        mother_2 = mask2 .* X(mother, :);
        father_1 = mask1 .* X(father, :);
        father_2 = mask2 .* X(father, :);
        X(parent_number + child, :) = mother_1 + father_2;
        X(parent_number+child+1, :) = mother_2 + father_1;
    end
    % mutation
    mutation_population = X(2:pop, :);
    number_of_elements = (pop - 1) * 3;
    number_of_mutations = ceil(number_of_elements * mutation_rate);
    mutation_points = ceil(number_of_elements * rand(1, number_of_mutations));
    mutation_population(mutation_points) = rand(1, number_of_mutations);
    X(2:pop, :) = mutation_population;

end

c = Gbest_position(1);
g = Gbest_position(2);
e = Gbest_position(3);
bestFitness = Gbest_fit;
cost = model;
save
end