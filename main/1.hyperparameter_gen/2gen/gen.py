import pygad
import numpy as np

# funcations
def goal_fitness_func(solution, solution_idx): # calculate solutions fitness
    output = np.abs(hyper-solution)
    fitness = 1.0 / (np.sum(output) + 0.000001)
    return fitness
last_fitness=0
def callback_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
    last_fitness = ga_instance.best_solution()[1]

# Goal
hyper=np.array([1, 2, 3, 4],dtype=float)

## Condition
# population
num_generations=10
sol_per_pop=100 # number of population in each generation
num_parents_mating=int(sol_per_pop/2) # how much parents will match
num_genes=len(hyper)

# condition of each gen
gene_space={'low': 1e-3, 'high': 10}, {'low': 1, 'high': 10}, {'low': 1, 'high': 10}, {'low': 1, 'high': 10}
gene_type=[float, float, float, float]

# technic
parent_selection_type = 'sss' # sss: steady-state, rws: roulette wheel, sus: stochastic universal, rank: rank, random: random, tournament: tournament
crossover_type="single_point" # single_point: single-point crossover(defult), two_points: two points crossover, uniform: uniform crossover, scattered: scattered crossover
mutation_type="swap" # random(defaults), swap, inversion, scramble, adaptive, None

fitness_func=goal_fitness_func
on_generation=callback_generation


seed=1 # None
parallel=['thread', 2]
save_best=False

# define gen model
ga_instance = pygad.GA(num_generations=num_generations,
                       sol_per_pop=sol_per_pop,
                       num_parents_mating=num_parents_mating,
                       mutation_type=mutation_type,
                       parent_selection_type =  parent_selection_type,
                       fitness_func=fitness_func,
                       num_genes=num_genes,
                       on_generation=callback_generation,
                       gene_space=gene_space,
                       gene_type=gene_type,
                       random_seed=seed,
                       parallel_processing=parallel,
                       save_best_solutions=save_best)

# Run
ga_instance.run()

# show results
# ga_instance.plot_fitness()

# print(len(ga_instance.population))
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))


'''
/opt/homebrew/Caskroom/miniforge/base/envs/torch3/lib/python3.9/site-packages/pygad/pygad.py:486: 
UserWarning: The percentage of genes to mutate (mutation_percent_genes=10) resutled in selecting (0) genes. 
The number of genes to mutate is set to 1 (mutation_num_genes=1).
If you do not want to mutate any gene, please set mutation_type=None.

if not self.suppress_warnings: warnings.warn("The percentage of genes to mutate (mutation_percent_genes={mutation_percent}) resutled in selecting ({mutation_num}) genes. 
The number of genes to mutate is set to 1 (mutation_num_genes=1).
If you do not want to mutate any gene, please set mutation_type=None.".format(mutation_percent=mutation_percent_genes, mutation_num=mutation_num_genes))
'''