#An edge list tree can be made using tuples for edges as items in a list. 
#E = (u, V) for trees has V the child and u the parent.  
_tree = [] #creates empty list 

def make_tree(u, V):        #function to add edges to the tree
    _tree.append((u, V))    #adds the edge as a tuple to the list

#initialize to fit example
make_tree(0, 1)             #adds edge between node 0 and node 1
make_tree(0, 2)             #node 0 and node 2 edge
make_tree(1, 3)             #node 1 and node 3 edge
make_tree(1, 4)             #node 1 and node 4 edge
make_tree(2, 5)             #node 2 and node 5 edge
print(_tree)