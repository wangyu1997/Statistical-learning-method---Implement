import numpy as np

class KdNode:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

class KdTree:
    def __init__(self, data=np.array([])):
        self.data = data
        self.size = len(data)
        self.N = len(data[0])
        self.kd_tree = self.build(0, self.size, 0)

    def build(self, left, right, index):
        if left >= right:
            return None
        elif right == left + 1:
            return KdNode(value=self.data[left])
        else:
            self.data[left:right] = \
                self.data[left:right][self.data[left:right, index].argsort()]
            mid = int((right + left) / 2)
            return KdNode(value=self.data[mid],
                          left=self.build(left, mid, (index + 1) % self.N),
                          right=self.build(mid + 1, right, (index + 1) % self.N))

    def mid_tra(self):
        self._mid_travel(self.kd_tree)

    def _mid_travel(self, node=None):
        if node is not None:
            print("value = ", node.value)
            self._mid_travel(node.left)
            self._mid_travel(node.right)

