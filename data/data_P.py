import pandas as pd
import os
from torch_geometric.data import Data 
import torch
import shutil
import sys
"""
Aquest codi és per lletgir les dades. Hi ha dues funcions principals; la primer
prendre les dades que necessitam i guardarles a data_processed; la segona llegeix
aquestes dades i les guarda en tres conjunts: training, validation i test.
"""

def process_raw_data():
    """
    Per que funcioni hem de tenir guardades les nostres dades en dos nivells superiors
    en una carpeta raw_data
    v TSG_SGNN
        > Capes
        > Entrenament
        > Model
        v data
            > data_P.py
    v raw_data
        >
        >
        ···
    
        RETURNS
    Aquesta funcio no torna cap objecte; sino que guarda les dades processades i 
    necessàries pel model en una carpeta processed_data a dos nivells superiors
    """

    ruta_actual = os.path.abspath(__file__)
    ruta_superior = os.path.dirname(os.path.dirname(os.path.dirname(ruta_actual)))
    
    
    destination = ruta_superior + "\\processed_data"
    os.makedirs(destination, exist_ok=True)
    os.makedirs(destination + '\\Individuals', exist_ok= True)

    organism_csv = pd.read_csv(ruta_superior+"\\raw_data\\data\\Results.csv", 
                               usecols=["organism","Categories"], 
                               na_values=["nan", "NaN", "NA", ""])
    """
    Com no podem treballar amb els valors NaN els hem d'eliminar
    """
    organism_csv = organism_csv.dropna()
    

    n =  organism_csv.shape[0]
    carpeta_destino = destination + '\\Individuals'
    ruta_org = ruta_superior + '\\raw_data\\data\\Individuals'

    doc_path = destination + '\\O.txt'
    doc_path2 = destination + '\\OK.csv'
    doc_path4 = destination + '\\Kn.csv'

    if os.path.exists(doc_path): os.remove(doc_path)
    if os.path.exists(doc_path2): os.remove(doc_path2)
    if os.path.exists(doc_path4): os.remove(doc_path4)

    doc = open(doc_path, 'w')
    organism_csv.to_csv(doc_path2, index= False)
    kingdoms = ['Animals', 'Bacteria','Archaea','Fungi','Plants','Protists']
    Kn = pd.DataFrame({
        "Kingdom": kingdoms,
        "NumData": [0, 0, 0, 0, 0, 0]
    })
    
    for i in range(n): 
        [organism, kingdom ] = organism_csv.iloc[i]
        doc.write(organism+'\n')
        arx = f'\\{organism}\\{organism}_R_adj.csv'
        ro = ruta_org + arx
        char = '='*int(100*i/n) + " "*(100-int(100*i/n))
        total = '[' + " "*100 + ']'
        sys.stdout.write(f"\r[{char}]")  # Escribe en la misma línea
        sys.stdout.flush()  # Forzar la actualización de la línea
        shutil.copy(ro, carpeta_destino) 
        Kn.loc[Kn["Kingdom"]==kingdom, "NumData"] += 1

    Kn.to_csv(doc_path4)
    f = " "*102
    sys.stdout.write(f"\r{f}")
    sys.stdout.write("\rData processed!")
    sys.stdout.flush()

def King2int(K):
    map = {'Animals': 0,
           'Bacteria': 1,
           'Archaea': 2,
           'Fungi': 3,
           'Plants': 4,
           'Protists': 5}
    return map[K]

def reading_data():
    """
    Aquesta funcio lletgeix les dades de individuals i les converteix en grafs.
    Retorna una llista dels grafs amb el seu regne animal (Kingdom)
    """
    ruta_actual = os.path.abspath(__file__)
    ruta_superior = os.path.dirname(os.path.dirname(os.path.dirname(ruta_actual)))
    
    ok_path = ruta_superior + '\\processed_data\\OK.csv'

    organism_csv = pd.read_csv(ok_path, 
                               usecols=["organism","Categories"])
    
    n =  organism_csv.shape[0]

    Graphs = []
    ruta = ruta_superior + '\\processed_data\\Individuals'
    for i in range(n): 
        [org, king] = organism_csv.iloc[i]

        O_ADJ_csv = pd.read_csv(ruta+f'\\{org}_R_adj.csv', sep=";", quotechar='"',
                                usecols=["source", "destination"])
        
        nodes = list(set(O_ADJ_csv['source'].to_list()).union(O_ADJ_csv['destination'].to_list()))
        map = {nodo: i for i, nodo in enumerate(nodes)}
        inverse_map = {i: nodo for i, nodo in enumerate(nodes)}

        origen = O_ADJ_csv['source'].map(map).to_list()
        destino = O_ADJ_csv['destination'].map(map).to_list()
        
        edge_index = torch.tensor([origen, destino], dtype=torch.long)
        x = nodes
        y = torch.tensor([King2int(king)],dtype=torch.long)

        G = Data(x=x, edge_index=edge_index, y=y)
        Graphs.append(G)

        char = '='*int(100*i/n) + " "*(100-int(100*i/n))
        sys.stdout.write(f"\r[{char}]")  # Escribe en la misma línea
        sys.stdout.flush()  # Forzar la actualización de la línea

    f = " "*102
    sys.stdout.write(f"\r{f}")
    sys.stdout.write(f"\rGraphs read correctly")
    sys.stdout.flush()
    return Graphs
    
def datasets():
    pass

process_raw_data()
reading_data()