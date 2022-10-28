import numpy as np
from itertools import combinations as comb
from copy import deepcopy

class Node:
    def __init__(self, val):
        self.l = None
        self.r = None
        self.v = val

class ree:
    """
    Tree where Nodes are False if they are minimal arguments
    """
    def __init__(self, OHencoding_length):
        self.root = Node(True)
        self.OHencoding_length = OHencoding_length
        self.nb_args = 0

    def getRoot(self):
        return self.root

    def encoded(self, arg):
        # arg is a tuple of indices where the OHencoded argument is 1
        # e.g. (0, 1, 2) for the argument [1, 1, 1, 0, 0, 0]
        arg_encoded = [0] * self.OHencoding_length
        for f in arg:
            arg_encoded[f] = 1
        return arg_encoded

    def add(self, arg):
        #print('Adding', arg)
        ismin = self.is_minimal(arg)
        if ismin:
            self.nb_args += 1
            #self._add(self.encoded(arg), self.root)
            self._add(arg, self.root)
            #print('Added', arg)

    def _add(self, arg, node):
        #assert (node.v is not False), "Not a minimal argument"
        if sum(arg):
            #assert (node.v is not False), "Not a minimal argument"
            if arg[-1]:
                if node.r is None:
                    if node.l is None:
                        node.r = Node(True)
                    else:
                        node.r = deepcopy(node.l) # risks of slow down
                self._add(arg[:-1], node.r)
            else:
                if node.l is None:
                    node.l = Node(True)
                self._add(arg[:-1], node.l)
                if node.r is not None:
                    # adding arg to right branch
                    if node.r.v:
                        self._add(arg[:-1], node.r)
        else:
            # Just popped the last 1
            node.v = False

    def is_minimal(self, arg):
        #return self._find(self.encoded(arg), self.root)
        return self._find(arg, self.root)

    def _find(self, arg, node):
        if node is None:
            return True
        if node.v is False:
            return False
        else:
            if sum(arg):
                if arg[-1]:
                    #left = self._find(arg.copy(), node.l)
                    #right = self._find(arg.copy(), node.r)
                    #return left and right
                    return self._find(arg[:-1], node.l) \
                        and self._find(arg[:-1], node.r)
                else:
                    return self._find(arg[:-1], node.l)
            else:
                return True
        
    def get_nb_args(self):
        return self.nb_args

    def printTree(self):
        if self.root is not None:
            self._printTree(self.root,'')

    def _printTree(self, node, prefix):
        if node is not None:
            self._printTree(node.r, prefix + ".")
            print(prefix, str(node.v)[0])
            self._printTree(node.l, prefix + ".")
            

    def potential_mins(self, instance, n):
        #return self._mins(instance, n, self.root)
        return self._mins(self.root, instance, n, [])

    def _mins(self, node, instance, n, current_path):
        if n == 0:
            if node is None:
                yield ([0]* len(instance) + current_path)
            elif node.v:
                yield ([0]* len(instance) + current_path)
        else:
            if node is None:
                #return combinations left 
                for c in comb(np.where(instance)[0], n):
                    prefix = [0] * len(instance)
                    for i in c:
                        prefix[i] = 1
                    yield (prefix + current_path)
            elif node.v:
                if instance[-1]:
                    if node.r is None:
                        yield from self._mins(node.l, instance[:-1], n-1, [1] + current_path)
                    else:
                        yield from self._mins(node.r, instance[:-1], n-1, [1] + current_path)
                yield from self._mins(node.l, instance[:-1], n, [0] + current_path)

"""
    def _mins(self, instance, n, node):
        print("n", n, instance)
        if node is None:
            if n == 0:
                # This potential arg is minimal
                if instance[-1]:
                    return [len(instance)]
                else:
                    return []
            else:
                # we can get all combinations of len n+1 of the remaing features
                # could do for[] yield
                return list(np.where(instance)[0]) #combinations len n?
        elif not node.v:
            # reached a minimal argument: current arg is not minimal
            return []
        else: # node.v is True
            if instance[-1]:
                if n == 0:
                    return [len(instance)]
                else:
                    if node.r is None:
                        #res = [len(instance)]
                        #res.append(self._mins(instance[:-1], n-1, node.l))
                        #return res
                        return [len(instance)] + [self._mins(instance[:-1], n-1, node.l)]
                    return self._mins(instance[:-1], n, node.l) +\
                             self._mins(instance[:-1], n-1, node.r)
            else:
                return self._mins(instance[:-1], n, node.l)
            if n == 0:
                # current arg is minimal
                # does this happen, if all args of len <n already exist? yes, 
                if instance[-1]:
                    return [len(instance)] + self._mins(instance[:-1], n-1, node.l)
                else:
                    return self._mins(instance[:-1], n, node.l)
            else:
                if instance[-1]:
                    # Forward check:
                    if node.r is None:
                        #res = [len(instance)]
                        #res.append(self._mins(instance[:-1], n-1, node.l))
                        #return res
                        return [len(instance)] + self._mins(instance[:-1], n-1, node.l)
                    return self._mins(instance[:-1], n, node.l) +\
                             self._mins(instance[:-1], n-1, node.r)
                else:
                    return self._mins(instance[:-1], n, node.l)
"""

def test():
    t = ree(8)
    t.printTree()
    t.add(t.encoded((1, 2)))
    t.printTree()
    t.add(t.encoded((3,)))
    t.printTree()
    assert(not t.is_minimal((3, 2)))
    assert(not t.is_minimal((1, 2, 4)))
    assert(t.is_minimal((0, 2)))
    print('--------#############--------')
    t.add(t.encoded((5,)))
    t.printTree()
    assert(not t.is_minimal((5, 6)))
    #print('tree:', t.root.l.r.v)
    #print('tree:', t.root.l.r.r.v)
    t.add(t.encoded((3, 6)))
    t.printTree()
    assert(not t.is_minimal((5, 6)))

def test2():
    t = ree(8)
    t.add(t.encoded((0,)))
    t.add(t.encoded((5,)))
    t.add(t.encoded((1,)))
    t.add(t.encoded((4,)))
    t.printTree()
    assert(not t.is_minimal(t.encoded((5, 6))))
    t.add(t.encoded((3, 6)))
    t.printTree()
    assert(not t.is_minimal(t.encoded((5, 6))))

def test3():
    t = ree(10)
    t.add(t.encoded((8,)))
    #t.add((0,2))
    #t.add((0,6))
    #t.add((4,6))
    t.add(t.encoded((4,7)))
    t.add(t.encoded((3,9)))
    t.add(t.encoded((1,5)))
    t.printTree()
    assert(not t.is_minimal(t.encoded((3,9))))

def test4():
    t = ree(4)
    t.add(t.encoded((3,1)))
    t.add(t.encoded((2,)))
    t.printTree()
    assert(not t.is_minimal(t.encoded((3,1))))

def test_mins():
    t = ree(7)
    #t.add((3,))
    ##t.add((0, 1, 2,4,5))
    #t.add((1, 4))
    t.add(t.encoded((1, 3)))
    t.add(t.encoded((1, 6)))
    t.add(t.encoded((2, 4)))
    t.printTree()

    instances = [[1, 0, 1, 0, 0, 1],[0, 1, 0, 1, 1, 0],[0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1]]

    arglen_range = range(0,4)
    generators = []
    for i in instances:
        g = []
        for n in arglen_range:
            g.append(t.potential_mins(i, n))
        generators.append(g)

    for i in range(len(instances)):
        for n in arglen_range:
            print('~', instances[i], np.where(instances[i])[0], 'n=', n)
            for arg in generators[i][n]:
                print(np.where(arg)[0], 'is a potential minimal arg:', t.is_minimal(arg))


#test()
#test2()
#test3()
#test4()
#test_mins()