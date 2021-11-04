import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, benchmarks

import random
from deap import tools

# ne pas oublier d'initialiser la graine aléatoire (le mieux étant de le faire dans le main))
random.seed()


def ea_simple(n, nbgen, evaluate, IND_SIZE, weights=(-1.0,)):
    """Algorithme evolutionniste elitiste

    Algorithme evolutionniste elitiste. 
    :param n: taille de la population
    :param nbgen: nombre de generation 
    :param evaluate: la fonction d'évaluation
    :param IND_SIZE: la taille d'un individu
    :param weights: les poids à utiliser pour la fitness (ici ce sera (-1.0,) pour une fonction à minimiser et (1.0,) pour une fonction à maximiser)
    """

    # fitness is a measure of quality of a solution
    creator.create("MaFitness", base.Fitness, weights=weights)
    # The first individual created will be a simple list containing floats
    creator.create("Individual", list, fitness=creator.MaFitness)

    toolbox = base.Toolbox()

    # Sélection des opérateurs de mutation, croisement, sélection avec des toolbox.register(...)
    toolbox.register("SBX", tools.cxSimulatedBinary, eta=15.0)
    toolbox.register("PBM", tools.mutPolynomialBounded, eta=15.0,low=-5,up=5,indpb=0.2)
    toolbox.register("select", tools.selBest)
    toolbox.register("func_init", random.uniform, -5, 5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.func_init, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", benchmarks.ackley)
    
    # Les statistiques permettant de récupérer les résultats
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # La structure qui permet de stocker les statistiques
    logbook = tools.Logbook()

    # La structure permettant de récupérer le meilleur individu
    hof = tools.HallOfFame(1)

    # Initialisation de la population en initialisant les individus
    pop = toolbox.population(n) 

    # Evaluation de la population à l'aide, dans notre cas, de la fonction ackley
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Applique pop à toutes les fonctions enregistrées et en retourne un dictionnaire
    record = stats.compile(pop)
    # Engistrement chronologique des données {'avg', 'std', 'max', 'min'} 
    logbook.record(gen=0,fit = fitnesses ,**record)
    
    p_co  = 0.5     # probability of cross-over 
    p_mu = 0.2      # probability of mutation
     
    for g in range(1,nbgen):

        # Selection de la progéniture
        Q = toolbox.select(pop,len(pop)) # this selects the best individuals in the population
        # Clone les individus sélectionnés
        Q = list(toolbox.map(toolbox.clone, Q)) #ensures we don’t own a reference to the individuals but an completely independent instance.

        # Applique un crossover et une mutation sur la progéniture
        for child1, child2 in zip(Q[::2],Q[1::2]):
            if random.random() < p_co:
                toolbox.SBX(child1,child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in Q:
            if random.random() < p_mu:
                toolbox.PBM(mutant)
                del mutant.fitness.values
                
        # Evaluation des nouveaux individus issus du crossover et de la mutation
        # qui possède une fitness invalide 
        invalid_ind = [ind for ind in Q if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
                
        pop[:] = Q
        """if (g%10==0):
            print("+",end="", flush=True)
        else:
            print(".",end="", flush=True)"""
        ## Mettre à jour les statistiques, le logbook et le hall-of-fame.
        record = stats.compile(pop)
        logbook.record(gen = g, **record)
        # Update the hall of fame with the population by replacing the worst
        hof.update(pop)
        
    return pop, hof, logbook