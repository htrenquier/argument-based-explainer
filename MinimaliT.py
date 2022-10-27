from msilib.schema import Error


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
        ismin = self.is_minimal(arg)
        if ismin:
            self.nb_args += 1
            self._add(self.encoded(arg), self.root)

    def _add(self, arg, node):
        assert node.v is not False, "Not a minimal argument"
        if sum(arg):
            if arg.pop():
                if node.r is None:
                    node.r = Node(True)
                self._add(arg, node.r)
            else:
                if node.l is None:
                    node.l = Node(True)
                self._add(arg, node.l)
        else:
            # Just popped the last 1
            node.v = False

    def is_minimal(self, arg):
        return self._find(self.encoded(arg), self.root)

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
            self._printTree(self.root, 1, '')

    def _printTree(self, node, count, prefix):
        if node is not None:
            self._printTree(node.l, count*2+1, 'l')
            print(prefix + ' ' + str(count) + str(node.v)[0])
            self._printTree(node.r , count*2, 'r')


def test():
    t = ree(8)
    t.printTree()
    t.add((1, 2))
    t.printTree()
    t.add((3,))
    t.printTree()
    assert(not t.is_minimal((3, 2)))
    assert(not t.is_minimal((1, 2, 4)))
    assert(t.is_minimal((0, 2)))
    print('--------#############--------')
    t.add((5,))
    t.printTree()
    assert(not t.is_minimal((5, 6)))
    #print('tree:', t.root.l.r.v)
    #print('tree:', t.root.l.r.r.v)
    t.add((3, 6))
    t.printTree()
    assert(not t.is_minimal((5, 6)))

def test2():
    t = ree(8)
    t.add((0,))
    t.add((5,))
    t.add((1,))
    t.add((4,))
    t.printTree()
    assert(not t.is_minimal((5, 6)))
    t.add((3, 6))
    t.printTree()
    assert(not t.is_minimal((5, 6)))

def test3():
    t = ree(10)
    t.add((8,))
    #t.add((0,2))
    #t.add((0,6))
    #t.add((4,6))
    t.add((4,7))
    t.add((3,9))
    t.add((1,5))
    t.printTree()
    assert(not t.is_minimal((3,9)))

def test4():
    t = ree(4)
    t.add((3,1))
    t.add((2,))
    t.printTree()
    assert(not t.is_minimal((3,1)))

#test()
#test2()
#test3()
#test4()