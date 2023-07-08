import numpy as np

class Chromosome():

    def __init__(self, x0, xmax, objfun,generation, pc=0.8, pm=0.1) -> None:
        self.x = np.array(x0)
        self.eval_value = self.Eval(objfun)
        self.crossover_p = pc
        self.mutation_p = pm
        self.mutation_range = np.array(xmax)
        self.generation = generation

    def Eval(self, objfun):
        return objfun(self.x)
    
    def Mutation(self):
        for i in range(len(self.x)):
            if np.random.rand() < self.mutation_p:
                self.x[i] = np.random.rand() * self.mutation_range[i]

def choose_individual(population):
    choosed_p = np.zeros(len(population))
    value_sum = 0
    for individual in population:
        value_sum += individual.eval_value
    for i,individual in enumerate(population):
        choosed_p[i] = individual.eval_value / value_sum
    rate = np.random.rand()
    m = 0
    for i in range(len(population)):
        m += choosed_p[i]
        if rate <= m:
            return i

def mate(population):
    mate_parts = []
    mate_index = []
    for i,individual in enumerate(population):
        if np.random.rand() < individual.crossover_p:
            mate_parts.append(individual)
            mate_index.append(i)
    if len(mate_parts) > 0:
        mate_parts_other = []
        n = len(mate_parts)-1
        for i in range(n):
            mate_parts_other.append(mate_parts[n-i])
        for i in range(n):
            change_bit = np.random.randint(len(mate_parts[i].x)-1)
            for j in range(len(mate_parts[i].x)):
                if j >= change_bit:
                    mate_parts_other[i].x[j] = mate_parts_other[i].x[j]
        for i in range(len(mate_index)):
            population[mate_index[i]] = mate_parts[i]
        return population
    else:
        return population
        
def GA(objfun, numbers,x0s, xmax: np.array, pc=0.8, pm=0.1, generations=10000):
    '''
        Parameters:
            objfun: the object function (max f(x))
            numbers: how many members of a generation
            x0s: the first generation point matrix, size: [numbers, dim]
            xmax: the maximum mutation step
            pc: the probability of mating. Default: 0.8
            pm: the probability of mutation. Default: 0.1
            generation: the number of iter generation. Default: 10000

        Returns:
            Best: the best solution. Type: np.array
            Best_value: the best optimized value. Type: np.float
            Best_generation: how many iter times we got the best answer. Type: int
            generation_iter_bestvalue: the best value of every times iter. Type: np.array.
    '''
    population = []
    generation_iter_bestvalue = np.zeros(generations)
    Best_generation = 0
    if len(x0s) != numbers:
        print('Dim of input x0 is not fit the numbers')
        return None, None, None, None
    for i,x0 in enumerate(x0s):
        individual = Chromosome(x0, xmax, objfun, 0, pc, pm)
        population.append(individual)
        if i == 0:
            Best = individual.x
            Best_value = individual.eval_value
        else:
            if individual.eval_value > Best_value:
                Best_value = individual.eval_value
                Best = individual.x.copy()
    generation_iter_bestvalue[0] = Best_value
    # begin GA
    for generation in range(generations):
        bechoosed = []
        for _ in range(len(population)):
            bechoosed.append(population[choose_individual(population)])
        population = bechoosed
        population = mate(population)
        for individual in population:
            individual.Mutation()
            individual.generation = generation
            individual.eval_value = individual.Eval(objfun)
            if individual.eval_value > Best_value:
                Best_value = individual.eval_value
                Best = individual.x.copy()
                Best_generation = generation
            generation_iter_bestvalue[generation] = Best_value
    return Best, Best_value, Best_generation, generation_iter_bestvalue