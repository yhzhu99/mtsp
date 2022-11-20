'''
The main helper class for Genetic Algorithm to perform
crossover, mutation on populations to evolve them
'''
from population import *

class GA:

    @classmethod
    # Evolve pop
    def evolvePopulation(cls, pop):

        newPopulation = Population(pop.populationSize, False)

        elitismOffset = 0
        # If fittest chromosome has to be passed directly to next generation
        if elitism:
            newPopulation.saveRoute(0, pop.getFittest())
            elitismOffset = 1

        # Performs tournament selection followed by crossover to generate child
        for i in range(elitismOffset, newPopulation.populationSize):
            parent1 = cls.tournamentSelection(pop)
            parent2 = cls.tournamentSelection(pop)
            child = cls.crossover(parent1, parent2)
            # Adds child to next generation
            newPopulation.saveRoute(i, child)


        # Performs Mutation
        for i in range(elitismOffset, newPopulation.populationSize):
            cls.mutate(newPopulation.getRoute(i))

        return newPopulation

    # Function to implement crossover operation
    @classmethod
    def crossover (cls, parent1, parent2):
        child = Route()
        child.base.append(Dustbin(-1, -1)) # since size is (numNodes - 1) by default
        startPos = 0
        endPos = 0
        while (startPos >= endPos):
            startPos = random.randint(1, numNodes-1)
            endPos = random.randint(1, numNodes-1)

        parent1.base = [parent1.route[0][0]]
        parent2.base = [parent2.route[0][0]]

        for i in range(numTrucks):
            for j in range(1, parent1.routeLengths[i]):
                parent1.base.append(parent1.route[i][j])


        for i in range(numTrucks):
            for j in range(1, parent2.routeLengths[i]):
                parent2.base.append(parent2.route[i][j])

        for i in range(1, numNodes):
            if i > startPos and i < endPos:
                child.base[i] = parent1.base[i]

        for i in range(numNodes):
            if not(child.containsDustbin(parent2.base[i])):
                for i1 in range(numNodes):
                    if child.base[i1].checkNull():
                        child.base[i1] =  parent2.base[i]
                        break

        k=0
        child.base.pop(0)
        for i in range(numTrucks):
            child.route[i].append(RouteManager.getDustbin(0)) # add same first node for each route
            for j in range(child.routeLengths[i]-1):
                child.route[i].append(child.base[k]) # add shuffled values for rest
                k+=1
        return child

    # Mutation opeeration
    @classmethod
    def mutate (cls, route):
        index1 = 0
        index2 = 0
        while index1 == index2:
            index1 = random.randint(0, numTrucks - 1)
            index2 = random.randint(0, numTrucks - 1)
        #print ('Indexes selected: ' + str(index1) + ',' + str(index2))

        #generate replacement range for 1
        route1startPos = 0
        route1lastPos = 0
        while route1startPos >= route1lastPos or route1startPos == 1:
            route1startPos = random.randint(1, route.routeLengths[index1] - 1)
            route1lastPos = random.randint(1, route.routeLengths[index1] - 1)

        #generate replacement range for 2
        route2startPos = 0
        route2lastPos = 0
        while route2startPos >= route2lastPos or route2startPos == 1:
            route2startPos = random.randint(1, route.routeLengths[index2] - 1)
            route2lastPos= random.randint(1, route.routeLengths[index2] - 1)

        #print ('startPos, lastPos: ' + str(route1startPos) + ',' + str(route1lastPos) + ',' + str(route2startPos) + ',' + str(route2lastPos))
        swap1 = [] # values from 1
        swap2 = [] # values from 2

        if random.randrange(1) < mutationRate:
            # pop all the values to be replaced
            for i in range(route1startPos, route1lastPos + 1):
                swap1.append(route.route[index1].pop(route1startPos))

            for i in range(route2startPos, route2lastPos + 1):
                swap2.append(route.route[index2].pop(route2startPos))

            del1 = (route1lastPos - route1startPos + 1)
            del2 = (route2lastPos - route2startPos + 1)

            # add to new location by pushing
            route.route[index1][route1startPos:route1startPos] = swap2
            route.route[index2][route2startPos:route2startPos] = swap1

            route.routeLengths[index1] = len(route.route[index1])
            route.routeLengths[index2] = len(route.route[index2])

    # Tournament Selection: choose a random set of chromosomes and find the fittest among them 
    @classmethod
    def tournamentSelection (cls, pop):
        tournament = Population(tournamentSize, False)

        for i in range(tournamentSize):
            randomInt = random.randint(0, pop.populationSize-1)
            tournament.saveRoute(i, pop.getRoute(randomInt))

        fittest = tournament.getFittest()
        return fittest
