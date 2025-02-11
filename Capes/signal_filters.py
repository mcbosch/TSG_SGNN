import numpy as np

'''
DUBTE: Per fer el codi d'una xarxa neuronal hem de fer una GNN
hem de definir un graf bipartit?¿?¿?¿

Per programar el model hauries de mirar SurveyGnn.pdf e intentar-ho 
fer en codi.
'''

class GraphConvFilter():

    '''
        We define our convolution with:
            > H is a vector of length K
        Thus the convolution of a signal x for the shift-operator S is:
            y = h0*x + h1*S*x + h2*S^2*x + ... + hK*S^K*x
        
        The parameters h0,h1,...,hK defines a weigh for the k-shifted
        signal S^k*x, and are learnable parameters. Note that x(i) is modifyed by S^k only for the 
        nodes that are k-hops away. 
    '''

    def __init__(self,H):
        '''
        Hem de tenir en compte que pentura un filtre depen de 
        '''
        self.order = len(H) - 1
        self.parameters = H

    def forward(self, S, x):
        K = self.order
        n = len(x)
        y = np.zeros(n,1)

        '''
         In real examples we have always sparse graphs; thus we should
         use a package which to calculate powers of a sparse matrix.
        '''
        
        S_anterior = np.identity(n)
        for i in range(K + 1):
            y = y + self.parameters[i]*np.matmul(S_anterior, x)
            S_anterior = np.matmul(S_anterior, S)
        
        return y

    def modify_par(self, H):
        self.__parameters = H 4
    
    def __repr__(self):
        return f'Convolutional layer of order {self.order} +\n 
            Parameters: {self.parameters}'
        

class ChebyshevPolynom():

    def __init__(self):
        pass