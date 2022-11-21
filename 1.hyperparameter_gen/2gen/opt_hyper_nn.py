import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from NN.nn import nnn

import pygad
import numpy as np

hyper=np.array([1, 2, 3, 4],dtype=float)
# hyper=np.array([learingrate, number_of_epoch, nodes_per_hidden, number_of_hidden])
# hyper=np.array([1e-2, 20, 10, 2],dtype=str)
# asd=nnn(hyper)


# function_inputs = np.array([0.1, 10, 10, 2])

def fitness_func(solution, solution_idx):

    abs_error = nnn(5, solution) + 0.00000001

    solution_fitness = 1.0 / abs_error # 학습이 너무 잘되서 변화가 없다는거 같음....

    return solution_fitness
fitness_function = fitness_func


last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    # print(ga_instance.population)
    # print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    # print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
    last_fitness = ga_instance.best_solution()[1]
    # print("*"*30)

ga_instance = pygad.GA(num_generations=20,
                       sol_per_pop=10,
                       num_parents_mating=5,
                       mutation_type='random',
                       parent_selection_type = "sss",
                       fitness_func=fitness_function,
                       num_genes=len(hyper),
                       on_generation=callback_generation,
                       gene_space=[{'low': 1e-4, 'high': 1e-2}, {'low': 1, 'high': 5}, {'low': 1, 'high': 2}, {'low': 1, 'high': 3}],
                       gene_type=[float, int, int, int])


# print(ga_instance.population)
ga_instance.run()

ga_instance.plot_fitness() 
# plot result
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))