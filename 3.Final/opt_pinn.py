import pygad
import numpy as np
import sys, os
from pinn import pinn
import time as T

seed=1114
np.random.seed(seed)

# hyper=np.array([1e-4,15000,32,3])
# learingrate, number_of_epoch, nodes_per_hidden, number_of_hidden
# asd=pinn(hyper=hyper, gif=False)
# print(asd)
learingrate_min, learingrate_max = 1e-4, 1e-1
number_of_epoch_min, number_of_epoch_max = 1, int(1e4)
nodes_per_hidden_min, nodes_per_hidden_max = 1, 20
number_of_hidden_min, number_of_hidden_max = 1, 20


equation_inputs=np.array([1, 2, 3, 4],dtype=float)


def fitness_func(solution, solution_idx):
    abs_error = pinn(solution, ga_instance.generations_completed, False) + 0.00000001
    solution_fitness = 1.0 / abs_error # 학습이 너무 잘되서 변화가 없다는거 같음....
    return solution_fitness

last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}"+"*********************************")
    print(ga_instance.population) # print population
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
    last_fitness = ga_instance.best_solution()[1]
    # print("*"*30)

## Conditions
# population
from numpy.random import uniform as uni
pop=20
initial_pop=np.array([
    uni(learingrate_min, learingrate_max, pop),
    uni(number_of_epoch_min, number_of_epoch_max, pop),
    uni(nodes_per_hidden_min, nodes_per_hidden_max, pop),
    uni(number_of_hidden_min, number_of_hidden_max, pop)]).T
num_generations=100000
# sol_per_pop=20#20 # number of population in generation
num_parents_mating=int(len(initial_pop)/2) # how much parents will match
# num_genes=len(equation_inputs)

# condition of each gen
gene_space=[{'low': learingrate_min, 'high': learingrate_max}, 
            {'low': number_of_epoch_min, 'high': number_of_epoch_max}, 
            {'low': nodes_per_hidden_min, 'high': nodes_per_hidden_max}, 
            {'low': number_of_hidden_min, 'high': number_of_hidden_max}]
# gene_type=[float, int, int, int]

# technic
parent_selection_type = 'sss' # sss: steady-state, rws: roulette wheel, sus: stochastic universal, rank: rank, random: random, tournament: tournament
crossover_type="single_point" # single_point: single-point crossover(defult), two_points: two points crossover, uniform: uniform crossover, scattered: scattered crossover
mutation_type="swap" # random(defaults), swap, inversion, scramble, adaptive, None
# mutation_by_replacement=False # bool parameter. it's only works when mutation_type="random". True: replace the gene by the randomly generated value. False: adding the random value to the gene.
# mutation_probability=1 # value of take mutaion. 0~1. to replace this parameters, parameters mutation_percent_genes and mutation_num_genes can be used.
mutation_percent_genes= 5# "default" # Percentage of genes to mutate. It defaults to the string "default" which is later translated into the integer 10 which means 10% of the genes will be mutated. It must be >0 and <=100. Out of this percentage, the number of genes to mutate is deduced which is assigned to the mutation_num_genes parameter. The mutation_percent_genes parameter has no action if mutation_probability or mutation_num_genes exist. Starting from PyGAD 2.2.2 and higher, this parameter has no action if mutation_type is None.
# mutation_num_genes=None # Number of genes to mutate which defaults to None meaning that no number is specified. this parameter has no action mutation_probability exists and mutation_type is None.
# random_mutation_min or max_val=-1.0 # For random mutation, the random_mutation_min_val parameter specifies the start and end value of the range from which a random value is selected to be added to the gene. It defaults to -1. Starting from PyGAD 2.2.2 and higher, this parameter has no action if mutation_type is None.

fitness_function = fitness_func
on_generation=callback_generation  #None callback_generation

parallel=20
save_best=False

s=T.time()
ga_instance = pygad.GA(initial_population=initial_pop,
                    num_generations=num_generations,
                    # sol_per_pop=sol_per_pop,
                    num_parents_mating=num_parents_mating,
                    mutation_type=mutation_type,
                    parent_selection_type =  parent_selection_type,
                    fitness_func=fitness_function,
                    # num_genes=num_genes,
                    on_generation=on_generation,
                    gene_space=gene_space,
                    # gene_type=gene_type,
                    random_seed=seed,
                    parallel_processing=parallel,
                    save_best_solutions=save_best)
# print(ga_instance.population)
ga_instance.run()
f=T.time()

# plot result
print("time: ",f-s)
# ga_instance.plot_fitness() 
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))