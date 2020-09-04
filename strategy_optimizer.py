## Imports
#from utils import data_fetch
from datetime import datetime
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
#import indicators
from db import timscale_setup
from db import dbscrape
import strategys_backtesting
from main import backtestModel

from backtesting import Backtest
#from backtesting.test import SMA
from backtesting import Strategy

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_surface
from pyswarms.utils.plotters.formatters import Mesher
## Pyswarms

class PyswarmOptimizer:
    def __init__(self, model=None):
        if not model:
            self.model = backtestModel()
            self.model.setData(self.model.fetchData(), verbose=True)
            self.model.prepareBacktest(strategys_backtesting.SmaCross)
        self.options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    def rosenbrock_with_args(self, x, a, b, c=0):
        f = (a - x[:, 0]) ** 2 + b * (x[:, 1] - x[:, 0] ** 2) ** 2 + c
        return f

    def optimize(self, objective_func=None):
        if not objective_func:
            objective_func = self.model.getCumReturnsError
        bounds = (np.array([10, 20]), np.array([50, 80]))
        self.optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=self.options, bounds=bounds)
        # Perform optimization
        print("Performing Optimization")
        #cost, pos = optimizer.optimize(fx.rastrigin, iters=1000)
        #cost, pos = optimizer.optimize(self.rosenbrock_with_args, 1000, a=1, b=100, c=0)
        cost, pos = optimizer.optimize(objective_func, iters=1000)
        print(" ".join(["Cost", str(cost), "Position", str(pos)]))

    def plot3d(self):
        # Prepare position history
        m = Mesher(func=self.model.getCumReturnsError)
        pos_history_3d = m.compute_history_3d(self.optimizer.pos_history)
        plot_surface(pos_history_3d)

## Bird Swarm Optimization

class PSO:
    w = 0.729
    c1 = 1.49445
    c2 = 1.49445
    lr = 0.01
    class Particle:
        def __init__(self, dim, minx, maxx):
            self.position = np.random.uniform(low=minx, high=maxx, size=dim)
            self.velocity = np.random.uniform(low=minx, high=maxx, size=dim)
            self.best_part_pos = self.position.copy()

            self.error = error(self.position)
            self.best_part_err = self.error.copy()

        def setPos(self, pos):
            self.position = pos
            self.error = error(pos)
            if self.error < self.best_part_err:
                self.best_part_err = self.error
                self.best_part_pos = pos
    
    def __init__(self, dims, numOfBoids, numOfEpochs):
        self.swarm_list = [self.Particle(dims, 10, 50) for i in range(numOfBoids)]
        self.numOfEpochs = numOfEpochs

        self.best_swarm_position = np.random.uniform(low=10, high=50, size=dims)
        self.best_swarm_error = 1e80  # Set high value to best swarm error
        self.model = backtestModel()
        self.model.setData(self.model.fetchData(), verbose=True)
        self.model.prepareBacktest(strategys_backtesting.SmaCross)

    def optimize(self):
        for i in range(self.numOfEpochs):

            for j in range(len(self.swarm_list)):

                current_particle = self.swarm_list[j]  # get current particle

                #Vcurr = grad_error(current_particle.position)  # calculate current velocity of the particle
                Vcurr = current_particle.velocity

                deltaV = self.w * Vcurr \
                        + self.c1 * (current_particle.best_part_pos - current_particle.position) \
                        + self.c2 * (self.best_swarm_position - current_particle.position)  # calculate delta V

                current_particle.velocity = Vcurr + deltaV
                new_position = self.swarm_list[j].position - self.lr * deltaV  # calculate the new position
                
                if new_position[0] <= 10:
                    new_position[0] = current_particle.position[0]
                if new_position[0] >= 50:
                    new_position[0] = current_particle.position[0]
                if new_position[1] <= 10:
                    new_position[1] = current_particle.position[1]
                if new_position[1] >= 50:
                    new_position[1] = current_particle.position[1]
                if (new_position[1] - new_position[0]) < 5:
                    new_position[1] = new_position[0] + 5
                

                self.swarm_list[j].setPos(new_position)  # update the position of particle

                current_error = self.model.getReturnsError(new_position)
                if current_error < self.best_swarm_error:  # check the position if it's best for swarm
                    self.best_swarm_position = new_position
                    self.best_swarm_error = current_error

            print('Epoch: {0} | Best position: [{1},{2}] | Best known error: {3}'.format(i,
                                                                                        self.best_swarm_position[0],
                                                                                        self.best_swarm_position[1],
                                                                                        self.best_swarm_error))




class Particle():
    def __init__(self):
        self.position = np.array([(-1) ** (bool(random.getrandbits(1))) * random.random()*50, (-1)**(bool(random.getrandbits(1))) * random.random()*50])
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = np.array([0,0])

    def __str__(self):
        print("I am at ", self.position, " meu pbest is ", self.pbest_position)
    
    def move(self):
        self.position = self.position + self.velocity


class Space():

    def __init__(self, target, target_error, n_particles):
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_position = np.array([random.random()*50, random.random()*50])
        self.W = 0.5
        self.c1 = 0.8
        self.c2 = 0.9

    def print_particles(self):
        for particle in self.particles:
            particle.__str__()
   
    def fitness(self, particle):
        return particle.position[0] ** 2 + particle.position[1] ** 2 + 1

    def set_pbest(self):
        for particle in self.particles:
            fitness_cadidate = self.fitness(particle)
            if(particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position
            

    def set_gbest(self):
        for particle in self.particles:
            best_fitness_cadidate = self.fitness(particle)
            if(self.gbest_value > best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = particle.position

    def move_particles(self):
        for particle in self.particles:
            new_velocity = (self.W*particle.velocity) + (self.c1*random.random()) * (particle.pbest_position - particle.position) + \
                            (random.random()*self.c2) * (self.gbest_position - particle.position)
            particle.velocity = new_velocity
            particle.move()
            
def optimize():
    n_iterations = 100
    target_error = 0.5
    n_particles = 10
    search_space = Space(1, target_error, n_particles)
    particles_vector = [Particle() for _ in range(search_space.n_particles)]
    search_space.particles = particles_vector
    search_space.print_particles()

    iteration = 0
    while(iteration < n_iterations):
        search_space.set_pbest()    
        search_space.set_gbest()

        if(abs(search_space.gbest_value - search_space.target) <= search_space.target_error):
            break

        search_space.move_particles()
        iteration += 1
        
    print("The best solution is: ", search_space.gbest_position, " in n_iterations: ", iteration)


"""
from pyswarm import pso

# Define the objective (to be minimize)
def returns(x, *args):
    bt, stats = strategys_backtesting.run_backtest(data_bt, strategys_backtesting.SmaCross, verbose=True)

def getData(banknifty_table, startdate = "2020-01-01 09:30:00", enddate = "2020-01-29 15:30:00"):
    data = dbscrape.gettablerange(*(timscale_setup.get_config()), banknifty_table, startdate, enddate)
    data_bt = strategys_backtesting.prepareData(data, verbose=True)
    return data_bt

def weight(x, *args):
    H, d, t = x
    B, rho, E, P = args
    return rho*2*np.pi*d*t*np.sqrt((B/2)**2 + H**2)

# Setup the constraint functions
def yield_stress(x, *args):
    H, d, t = x
    B, rho, E, P = args
    return (P*np.sqrt((B/2)**2 + H**2))/(2*t*np.pi*d*H)

def buckling_stress(x, *args):
    H, d, t = x
    B, rho, E, P = args
    return (np.pi**2*E*(d**2 + t**2))/(8*((B/2)**2 + H**2))

def deflection(x, *args):
    H, d, t = x
    B, rho, E, P = args
    return (P*np.sqrt((B/2)**2 + H**2)**3)/(2*t*np.pi*d*H**2*E)

def constraints(x, *args):
    strs = yield_stress(x, *args)
    buck = buckling_stress(x, *args)
    defl = deflection(x, *args)
    return [100 - strs, buck - strs, 0.25 - defl]

# Define the other parameters
B = 60  # inches
rho = 0.3  # lb/in^3
E = 30000  # kpsi (1000-psi)
P = 66  # kip (1000-lbs, force)
args = (B, rho, E, P)

# Define the lower and upper bounds for H, d, t, respectively
lb = [10, 1, 0.01]
ub = [30, 3, 0.25]

xopt, fopt = pso(weight, lb, ub, args=args)
"""

## Search Algos

def GridSearch():
    model = RandomForestRegressor(n_jobs=-1, random_state=42, verbose=2)
 
    grid = {'n_estimators': [10, 13, 18, 25, 33, 45, 60, 81, 110, 148, 200],
            'max_features': [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25],
            'min_samples_split': [2, 3, 5, 8, 13, 20, 32, 50, 80, 126, 200]}
    
    rf_gridsearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=4, 
            cv=cv, verbose=2, return_train_score=True)
    
    rf_gridsearch.fit(X1, y1)
    
    # after several hours
    df_gridsearch = pd.DataFrame(rf_gridsearch.cv_results_)

 
def distance(x1, y1, x2, y2):
    dist = math.pow(x2-x1, 2) + math.pow(y2-y1, 2)
    return dist
 
def sumOfDistances(x1, y1, px1, py1, px2, py2, px3, py3, px4, py4):
    d1 = distance(x1, y1, px1, py1)
    d2 = distance(x1, y1, px2, py2)
    d3 = distance(x1, y1, px3, py3)
    d4 = distance(x1, y1, px4, py4)
 
    return d1 + d2 + d3 + d4
 
def newDistance(x1, y1, point1, point2, point3, point4):
    d1 = [x1, y1]
    d1temp = sumOfDistances(x1, y1, point1[0],point1[1], point2[0],point2[1],
                                point3[0],point3[1], point4[0],point4[1] )
    d1.append(d1temp)
    return d1

def newPoints(minimum, d1, d2, d3, d4):
    if d1[2] == minimum:
        return [d1[0], d1[1]]
    elif d2[2] == minimum:
        return [d2[0], d2[1]]
    elif d3[2] == minimum:
        return [d3[0], d3[1]]
    elif d4[2] == minimum:
        return [d4[0], d4[1]]

def HillClimbing():
    increment = 0.1
    startingPoint = [1, 1]
    point1 = [1,5]
    point2 = [6,4]
    point3 = [5,2]
    point4 = [2,1]

    minDistance = sumOfDistances(startingPoint[0], startingPoint[1], point1[0],point1[1], point2[0],point2[1], point3[0],point3[1], point4[0],point4[1] )
    flag = True
    i = 1
    while flag:
        d1 = newDistance(startingPoint[0]+increment, startingPoint[1], point1, point2, point3, point4)
        d2 = newDistance(startingPoint[0]-increment, startingPoint[1], point1, point2, point3, point4)
        d3 = newDistance(startingPoint[0], startingPoint[1]+increment, point1, point2, point3, point4)
        d4 = newDistance(startingPoint[0], startingPoint[1]-increment, point1, point2, point3, point4)
        print (i,' ', round(startingPoint[0], 2), round(startingPoint[1], 2))
        minimum = min(d1[2], d2[2], d3[2], d4[2])
        if minimum < minDistance:
            startingPoint = newPoints(minimum, d1, d2, d3, d4)
            minDistance = minimum
            #print i,' ', round(startingPoint[0], 2), round(startingPoint[1], 2)
            i+=1
        else:
            flag = False

