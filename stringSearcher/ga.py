####
##  uses Genetic Algorithm to find optimal solution
##  chromosome consists of ASCII char between 'A' and 'z'
##  parameters can be modified in main function
####

import random
import time
import itertools
import matplotlib.pyplot as plt


def initializePopulation(init_pop, chrom_len):
    """
    initialize population with random chromosomes
    :param init_pop: (int) population size
    :param chrom_len: (int) length of chromosomes
    :return: (list) list of chromosomes, randomly initialized
    """
    # initial population is always 32
    population = [[] for _ in range(init_pop)]

    # generate chromosomes randomly
    for i in range(init_pop):
        population[i] = random.sample(range(min_val, max_val+1), chrom_len)

    return population


def printPopulation(population):
    """
    prints population with fitness of each chromosome
    """
    output = ""
    for chromosome in population:
        tmp = [chr(allele) for allele in chromosome]
        for char in tmp:
            output += char
        fitness = fitnessFunction(chromosome)
        output += "\tF: " + str(fitness) + "\n"
    print(output)


def plotSearchSpace():
    chromosomes = itertools.product(list(range(min_val, max_val+1)), repeat=chromosome_len)
    chromosomes = [list(chrom) for chrom in chromosomes]
    
    fitness_dist = [fitnessFunction(chrom, set_method=True, dist=True) for chrom in chromosomes]
    fitness_bin = [fitnessFunction(chrom, set_method=True, dist=False) for chrom in chromosomes]
    
    plt.figure(figsize=(12,4), dpi=100)
    fig, (x1,x2) = plt.subplots(2, sharex=True)
    x1.plot(fitness_dist)
    x1.set_title("ASCII value distance for fitness")
    x1.set(ylabel="fitness")
    x2.plot(fitness_bin)
    x2.set_title("Binary decision for fitness")
    x2.set(ylabel="fitness", xlabel="chromosome in lexicographical order")
    fig.savefig("searchSpace.png")


def nextGen(population):
    """
    find the next generation of given population
    pick top half chromosomes -> mate with N-point crossover -> mutation
    :param population: (list) population to update
    """
    next_pop = [[] for _ in range(int(initial_pop/2))]
    half_pop = int(len(next_pop)/2)     # population size is assumed to be even
    
    # get top 8 (or half of the population) chromosomes
    score = []
    for chrom in population:
        score.append(fitnessFunction(chrom))
    sorted_score = [i[0] for i in sorted(enumerate(score), key=lambda x:x[1])]
    for i in range(half_pop):
        next_pop[i] = population[sorted_score[i]]
    
    # mate between even pairs and odd pairs
    for i in range(int(half_pop/2)):
        if not randomly:
            next_pop[half_pop+i*2], next_pop[half_pop+i*2+1] = mate(next_pop[i*2],next_pop[i*2+1], cross_pt)
        else:
            next_pop[half_pop+i*2] = random.sample(range(min_val, max_val+1), chromosome_len)
            next_pop[half_pop+i*2+1] = random.sample(range(min_val, max_val+1), chromosome_len)

    # mutation
    for i, chrom in enumerate(next_pop):
        next_pop[i] = [mutate(allele) for allele in chrom]

    return next_pop, 0 in score


def mate(parent1, parent2, N=1):
    """
    mate using N-point crossover
    default = 1-point crossover
    alleles are ascii values, not characters
    :param parent1, parent2: (list) chromosome
    :param N: 1 <= N < len(chromosome)
    :return: tuple of 2 child chromosomes
    """
    parents = [parent1, parent2]
    child1 = []
    child2 = []
    if N == 1:
        point = random.randint(1, len(parent1)-1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
    else:
        points = list(range(1,len(parent1)))
        random.shuffle(points)
        points = points[:N]
        points.sort()
        for i in range(N):
            j = i%2
            if i == 0:
                child1.extend(parents[0][:points[i]])
                child2.extend(parents[1][:points[i]])
            elif i == N-1:
                child1.extend(parents[j][points[i-1]:points[i]] + parents[1-j][points[i]:])
                child2.extend(parents[1-j][points[i-1]:points[i]] + parents[j][points[i]:])
            else:
                child1.extend(parents[j][points[i-1]:points[i]])
                child2.extend(parents[1-j][points[i-1]:points[i]])
    return child1, child2


def mutate(allele):
    """
    decides if to mutate given allele
    mutation operator: randomly pick a valid char
    :param allele: (int) ascii value of an allele
    :return: (int) mutated (or not) allele
    """
    mut = random.choices([False,True], weights=(1-mutation_rate, mutation_rate), k=1)
    return random.randint(min_val,max_val) if mut[0] else allele


def fitnessFunction(chromosome, set_method=False, dist=True):
    """
    calculates the fitness of population based on given optimal
    the smaller the fitness the better, optimal fitness is 0
    type of fitness function depends on defined global variable
        set 'distance' to True to use distance of ASCII value
        set 'distance' to False to use binary decision
    :param chromosome: (list) a list of ASCII values
    :return: (int) fitness of given chromosome
    """
    fitness = 0
    if (distance and not set_method) or (set_method and dist):
        # uses distance of ASCII
        for i, allele in enumerate(chromosome):
            fitness += (allele - optimal_ascii[i])**2
    else:
        # uses binary decision
        for i, allele in enumerate(chromosome):
            fitness +=  0 if allele==optimal_ascii[i] else 1
    return fitness


if __name__ == "__main__":
    
    # defined here to be used as global variables
    # min and max possible values of allele
    min_val = ord('A')
    max_val = ord('z')
    
    # default parameters in PART I but experiment for PART II
    initial_pop = 32                        # initial population size, power of 2
    optimal_str = "IowaState_Cyclones"      # optimal solution
    optimal_ascii = [ ord(char) for char in optimal_str ]
    chromosome_len = len(optimal_ascii)
    cross_pt = 1                            # for N-point crossover, 1 <= N < length of solution
    
    # experiment for PART I
    mutation_rate = 0.03                    # mutation rate, recommend 0.003 for ASCII distance fitness
    distance = False                        # set to False to use binary decision for fitness
    randomly = False                        # set to True to experiment with random generation
    max_gen = 500000                        # maximum number of generations allowed

#    # uncomment this chunk to test with random string
#    chromosome_len = 50                    # length of optimal solution
#    optimal_ascii = random.sample(range(min_val, max_val+1), chromosome_len)
#    tmp = [chr(allele) for allele in optimal_ascii]
#    optimal_str = ""
#    for char in tmp:
#        optimal_str += char

    # initializing population...
    print("Initial population")
    pop = initializePopulation(initial_pop, chromosome_len)
    printPopulation(pop)

    # running algorithm...
    start = time.time()
    i = 0
    found = False
    while not found and i < max_gen:
        print("\nGeneration " + str(i+1))
        print("Optimum solution: " + optimal_str)
        pop, found = nextGen(pop)
        printPopulation(pop)
        i += 1
    end = time.time()

    # output stats...
    print("\ntime taken (s): " + str(end-start))
    print("Initial population size: " + str(initial_pop))
    print("Recombination: " + str(cross_pt) + "-point crossover")
    print("Mutation rate: " + str(mutation_rate))
    print("Fitness function: " + ("ASCII distance" if distance else "binary decision") + "\n")


