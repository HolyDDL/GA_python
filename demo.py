import matplotlib.pyplot as plt
import numpy as np
import ga 
def objfun(xs):
    mother = 1.0
    for x in xs:
        mother += x*x
    return 1/mother
x0s = 10*np.random.rand(15,4)
xmax = np.ones(4)
Best, Best_value, Best_generation, generation_iter = ga.GA(objfun, len(x0s), x0s, xmax)
print(f'The best solution is {Best}')
print(f'The maximum value is {Best_value}')
print(f'In the {Best_generation}th generation got the best')
times = range(len(generation_iter))
plt.plot(times, generation_iter)
plt.xlabel('iter generation')
plt.ylabel('best value')
plt.show()