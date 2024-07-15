# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount= 0.9, iterations= 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        #print(self.iterations)
        # print(iterations)
    
        for k in range(100):
            states = self.mdp.getStates() # seleciona todos os estados
            newStatesValues = util.Counter()

            for state in states: # percorre cada estado
                if not self.mdp.isTerminal(state):   # se não for um estado terminal
                    # print(state)
                    actions = self.mdp.getPossibleActions(state) # seleciona todas as ações possiveis deste estado
                    #print(actions)
                    actionsValues = util.Counter()

                    for action in actions:  # percorre cada ação
                        transitionsAndProbs = self.mdp.getTransitionStatesAndProbs(state, action) # seleciona os pares transição-probabilidade
                        transitionValues = 0
                        #print(transitionsAndProbs)

                        for transition in transitionsAndProbs: # percorre cada transição (cim, baixo, esq, dir)
                            # print(self.mdp.getReward(state, action, transition[0]))
                            #print(transition)
                            #print(transition[1])
                            #print(transition[0])
                            #print(f"recompensa {self.mdp.getReward(transition[0], action, transition[0])}")
                            transitionValues += transition[1] * self.mdp.getReward(transition[0], action, transition[0]) * self.discount # calcula os valores 
                        #print(f'transicao: {transitionValues}')
                        # print(".")
                        actionsValues[action] = transitionValues 

                    print(actionsValues)
                    bestAction = actionsValues.__getitem__(actionsValues.argMax()) # seleciona o valor da ação de maior valor
                    print(bestAction)
                    print(self.mdp.getReward(state, action, state))
                    newStatesValues[state] = bestAction +  self.mdp.getReward(state, action, state) # recompensa imediata
            
            self.values = newStatesValues.copy()
            #print(self.values)
  
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # stateValue = self.values[state]
        transitionsAndProbs = self.mdp.getTransitionStatesAndProbs(state, action) # retorna os pares estado-probabilidade em uma lista
        qValue = 0

        for transition in transitionsAndProbs: # percorre cada transição (cim, baixo, esq, dir)
            qValue += transition[1] * self.values[transition[0]]  # calcula os valores 
                        
        return qValue
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        else:
            actions = self.mdp.getPossibleActions(state) # seleciona todas as ações possiveis deste estado
            actionsValues = util.Counter()

            for action in actions:  # percorre cada ação
                transitionsAndProbs = self.mdp.getTransitionStatesAndProbs(state, action) # seleciona os pares transição-probabilidade
                transitionValues = 0

                for transition in transitionsAndProbs: # percorre cada transição (cim, baixo, esq, dir)
                    transitionValues += transition[1] * self.values[transition[0]] # calcula os valores 
                        
                actionsValues[action] = transitionValues # soma os valores de todas as transições

            return actionsValues.argMax()
            
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
