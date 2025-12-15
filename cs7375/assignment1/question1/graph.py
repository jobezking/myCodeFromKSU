#Use dictionary to represent adjacency list graph implementation defined by G=(V,E) and E=(u,V)
G = {}   #create empty graph dictionary

def _V(G, V):       # function to create node
    if V not in G:  # check to see if node already exists
        G[V] = []   #if not, add the node as a dictonary key with an empty list to hold edges if they are added later

def _E(G, u, V, directed="True"): #function to add edges to nodes
    #add vertices if they are not already there
    _V(G, u)    #add node u
    _V(G, V)    #add node V
    #add edge
    G[u].append(V) #add edge between u and V

    if not directed:    #if undirected, add edge in oppsite direction
        G[V].append(u)              #add edge in oppsite direction between V and u

#The graph example is a complete K3 graph, making it undirected. It is initialized as follows:
_E(G, 'A', 'B', False)
_E(G, 'A', 'C', False)
_E(G, 'B', 'C', False)

print(G) #prints the graph