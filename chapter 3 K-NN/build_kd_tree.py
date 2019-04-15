import numpy as np


class KdNode:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.parent = None
        self.visited = False

    def clean_visit(self):
        self.visited = False


class KdTree:
    def __init__(self, data=np.array([])):
        self.data = data
        self.size = len(data)
        self.N = len(data[0])
        self.kd_tree = self.build(0, self.size, 0)

    def build(self, left, right, index, parent=None):
        if left >= right:
            return None
        elif right == left + 1:
            return KdNode(value=self.data[left])
        else:
            self.data[left:right] = \
                self.data[left:right][self.data[left:right, index].argsort()]
            mid = int((right + left) / 2)
            node = KdNode(value=self.data[mid])
            node.parent = parent
            node.left = self.build(left, mid, (index + 1) % self.N, node)
            node.right = self.build(mid + 1, right, (index + 1) % self.N, node)
            return node

    def mid_tra(self):
        self._mid_travel(travel=lambda node: print("value = ", node.value), node=self.kd_tree)

    def _mid_travel(self, travel, node=None):
        if node is not None:
            travel(node)
            self._mid_travel(travel, node.left)
            self._mid_travel(travel, node.right)

    def search_nearest(self, curr_node, value, distance):
        dis = distance(value, curr_node.value)
        curr_node.visited = True
        if curr_node.parent is not None:
            p_dis = distance(value, curr_node.parent.value)

    def find_leaf_node(self, value=None):
        return self._find_leaf_node(self.kd_tree, search_node=KdNode(value=value), index=0)

    def _find_leaf_node(self, curr_node, search_node, index):
        search_value = search_node.value
        if curr_node.left is None:
            return curr_node
        if search_value[index] <= curr_node.value[index]:
            return self._find_leaf_node(curr_node.left, search_node, (index + 1) % self.N)
        elif curr_node.right is None:
            return curr_node
        else:
            return curr_node.right

    def clean_visited_state(self):
        self._mid_travel(travel=lambda node: node.clean_visit(node), node=self.kd_tree)


data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
tree = KdTree(data)
tree.mid_tra()
print(tree.find_leaf_node([8.1, 9]).value)
