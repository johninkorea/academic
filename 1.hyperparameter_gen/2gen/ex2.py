import pygad
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

seed=1114
np.random.seed(seed)

## funcations
def goal_fitness_func(solution, solution_idx): # calculate solutions fitness
    output = np.power(hyper-solution,2)
    fitness = 1.0 / (np.sum(output) + 0.000001)
    return fitness
last_fitness=0
def callback_generation(ga_instance):
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}"+'***********************')
    # print(f"Fitness    = {ga_instance.best_solution()[1]}")
    # print(f"Change     = {ga_instance.best_solution()[1] - last_fitness}")
    print(ga_instance.population) # 세대즐 보기
    data.append(ga_instance.best_solution()[0])
    # fittness.append(ga_instance.best_solution()[1])
    fittness.append(ga_instance.best_solution()[1] - last_fitness)
    last_fitness = ga_instance.best_solution()[1]

# Goal
hyper=np.array([1, 2, 3, 4],dtype=float)

## Conditions
# population
initial_pop=np.random.uniform(0,5,20).reshape(5,4)
print(initial_pop)


num_generations=650
# sol_per_pop=5 # number of population in generation
# num_parents_mating=int(sol_per_pop/2) # how much parents will match
num_parents_mating=int(len(initial_pop)/2) # how much parents will match
# num_genes=len(hyper)

# condition of each gen
gene_space=[{'low': 0, 'high': 5}, {'low': 0, 'high': 5}, {'low': 0, 'high': 5}, {'low': 0, 'high': 5}]
# gene_type=[float, float, float, float]

# technic
parent_selection_type = 'sss' # sss: steady-state, rws: roulette wheel, sus: stochastic universal, rank: rank, random: random, tournament: tournament
crossover_type="single_point" # single_point: single-point crossover(defult), two_points: two points crossover, uniform: uniform crossover, scattered: scattered crossover
mutation_type="random" # random(defaults), swap, inversion, scramble, adaptive, None
# mutation_by_replacement=False # bool parameter. it's only works when mutation_type="random". True: replace the gene by the randomly generated value. False: adding the random value to the gene.
# mutation_probability=1 # value of take mutaion. 0~1. to replace this parameters, parameters mutation_percent_genes and mutation_num_genes can be used.
mutation_percent_genes= 5# "default" # Percentage of genes to mutate. It defaults to the string "default" which is later translated into the integer 10 which means 10% of the genes will be mutated. It must be >0 and <=100. Out of this percentage, the number of genes to mutate is deduced which is assigned to the mutation_num_genes parameter. The mutation_percent_genes parameter has no action if mutation_probability or mutation_num_genes exist. Starting from PyGAD 2.2.2 and higher, this parameter has no action if mutation_type is None.
# mutation_num_genes=None # Number of genes to mutate which defaults to None meaning that no number is specified. this parameter has no action mutation_probability exists and mutation_type is None.
# random_mutation_min or max_val=-1.0 # For random mutation, the random_mutation_min_val parameter specifies the start and end value of the range from which a random value is selected to be added to the gene. It defaults to -1. Starting from PyGAD 2.2.2 and higher, this parameter has no action if mutation_type is None.

fitness_func=goal_fitness_func
on_generation=callback_generation  #None callback_generation

seed=seed # None 1114
parallel=['thread', 2]
save_best=False

data=[]
fittness=[]

## define gen model
ga_instance = pygad.GA(initial_population=initial_pop,
                    num_generations=num_generations,
                    # sol_per_pop=sol_per_pop,
                    num_parents_mating=num_parents_mating,
                    mutation_type=mutation_type,
                    parent_selection_type =  parent_selection_type,
                    fitness_func=fitness_func,
                    # num_genes=num_genes,
                    on_generation=on_generation,
                    gene_space=gene_space,
                    # gene_type=gene_type,
                    random_seed=seed,
                    parallel_processing=parallel,
                    save_best_solutions=save_best)

# Run
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
## show results
# ga_instance.plot_fitness()
# print(len(ga_instance.population))
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")
if ga_instance.best_solution_generation != -1:
    print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")


# gif 만들기
import os
os.system("rm -rf img")
os.system("mkdir img")

def save_gif_PIL(outfile, files, fps=5, loop=0): # 유전들이 근접해가는 거ㄹ를 그려보자
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

files = []
fit=0
z=0
while z<len(data):
    if fittness[z]>0:
        # print(fit)
        file = f"img/nn_{z}.png"
        files.append(file)

        mse=np.power(np.mean(hyper-data[z]), 2)
        print(mse)

        plt.axis([0,5,0,5])
        plt.grid()
        
        plt.scatter(hyper, hyper, marker="s", c='r', label="Goal")
        plt.scatter(hyper, data[z], marker="*", alpha=1,c='b',label="Best Solution")
        
        plt.text(.5, 4.5, f"{z} ", size=15)
        plt.text(.95, 4.5, f"Generation", size=15)
        # plt.text(2.5, 4.5, f"MSE = {round(mse,5)}", size=10)
        
        plt.ylabel("value", size=15)
        plt.xlabel("gene index", size=15)
        plt.legend(loc='lower right')
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="white")
        plt.cla()
        fit=fittness[z]
    z+=1

z=0
while z<8:
    files.append(files[-1])    
    z+=1

save_gif_PIL("img/ggg.gif", files, fps=2, loop=0)
