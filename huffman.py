from queue import PriorityQueue


class HuffmanTree:

    class __Node:

        def __init__(self, value, freq, left_child, right_child):
            self.value = value
            self.freq = freq
            self.left_child = left_child
            self.right_child = right_child
        
        # classmethod will implicitly contain the object, updated the object with value and freq.
        @classmethod
        def init_leaf(self, value, freq):
            return self(value, freq, None, None)
        
        # It'll add the frequencies of left child and right child, value is initalized to None
        @classmethod
        def init_node(self, left_child, right_child):
            freq = left_child.freq + right_child.freq
            return self(None, freq, left_child, right_child)

        # Nodes with Values are only leaf nodes, as rest added as None
        def is_leaf(self):
            return self.value is not None

        def __eq__(self, other):
            stup = self.value, self.freq, self.left_child, self.right_child
            otup = other.value, other.freq, other.left_child, other.right_child
            return stup == otup

        def __nq__(self, other):
            return not (self == other)

        def __lt__(self, other):
            return self.freq < other.freq

        def __le__(self, other):
            return self.freq < other.freq or self.freq == other.freq

        def __gt__(self, other):
            return not (self <= other)

        def __ge__(self, other):
            return not (self < other)

    def __init__(self, arr):

        # ascending order priority queue
        q = PriorityQueue()

        # calculate frequencies and insert them into a priority queue
        # acquired dictionary of (element, fequency) from __calc_freq() method
        for val, freq in self.__calc_freq(arr).items():
            q.put(self.__Node.init_leaf(val, freq))

        
        while q.qsize() >= 2:
            # Popping two minimum nodes 
            u = q.get()
            v = q.get()

            # Constructed new node from minimum nodes
            q.put(self.__Node.init_node(u, v))

        # Storing the root node
        self.__root = q.get()

        # dictionaries to store huffmann table
        self.__value_to_bitstring = dict()

    # Returns huffmann table dictionary
    def value_to_bitstring_table(self):
        if len(self.__value_to_bitstring.keys()) == 0:
            self.__create_huffman_table()
        return self.__value_to_bitstring

    # Created dictionary for each key containing its bit representation as value
    def __create_huffman_table(self):
        def tree_traverse(current_node, bitstring=''):
            if current_node is None:
                return
            if current_node.is_leaf():
                self.__value_to_bitstring[current_node.value] = bitstring
                return
            tree_traverse(current_node.left_child, bitstring + '0')
            tree_traverse(current_node.right_child, bitstring + '1')

        tree_traverse(self.__root)

    # Generating dictionary of (element, frequency)
    def __calc_freq(self, arr):
        freq_dict = dict()
        for elem in arr:
            if elem in freq_dict:
                freq_dict[elem] += 1
            else:
                freq_dict[elem] = 1
        return freq_dict
