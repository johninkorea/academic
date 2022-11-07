import pygad
import numpy
hyper=numpy.array([1, 2, 3, 4],dtype=float)


function_inputs = numpy.random.random(4)
desired_output = 0

def fitness_func(solution, solution_idx):
    output = numpy.abs(hyper-solution)
    fitness = 1.0 / (numpy.sum(output) + 0.000001)
    return fitness

fitness_function = fitness_func

def callback_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
    last_fitness = ga_instance.best_solution()[1]

ga_instance = pygad.GA(num_generations=300,
                       num_parents_mating=5,
                       fitness_func=fitness_function,
                       sol_per_pop=10,
                       num_genes=len(function_inputs),
                       on_generation=callback_generation,
                       gene_type=[int, float, int, float])

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()

print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))






import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from NN.nn import nnn

import numpy as np


# hyper=np.array([learingrate, number_of_epoch, nodes_per_hidden, number_of_hidden])
hyper=np.array([1e-2, 20, 10, 2],dtype=str)
asd=nnn(hyper)

# 불러 학습하기 성공