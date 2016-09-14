class Unit:
    def __init__(self, *parents):
        self.parents = list(parents)
        for parent in parents:
            parent.add_child(self)

        self.children = []

    def add_parent(self, parent):
        self.parents.append(parent)
        parent.add_child(self)

    def get_parents(self):
        return self.parents

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def __add__(self, other):
        return Sum(self, other)

    def __mul__(self, other):
        return Product(self, other)


class Variable(Unit):
    def __init__(self, value=None):
        self.value = value
        super().__init__()

    def evaluate(self):
        return self.value

    def __add__(self, other):
        if self.evaluate() == 0:
            return other
        return super().__add__(other)

    def __mul__(self, other):
        if self.evaluate() == 1:
            return other
        return super().__mul__(other)


class Sum(Unit):
    def evaluate(self):
        return sum(parent.evaluate() for parent in self.parents)

    def get_gradient(self, parent):
        return Variable(1)

    def __iadd__(self, other):
        self.add_parent(other)
        return self


class Product(Unit):
    def evaluate(self):
        result = 1.0
        for parent in self.parents:
            result *= parent.evaluate()
        return result

    def get_gradient(self, parent):
        if len(self.parents) == 1:
            return Variable(1.0)
        elif len(self.parents) == 2:
            if parent is self.parents[0]:
                return self.parents[1]
            else:
                return self.parents[0]
        else:
            return Product(*(p for p in self.parents if p is not parent))

    def __mul__(self, other):
        self.add_parent(other)
        return self


def differentiate(target, variable):
    if variable is target:
        return Variable(1)

    gradient = Variable(0)

    for child in variable.get_children():
        local_gradient = child.get_gradient(variable)
        child_gradient = differentiate(target, child)
        gradient += child_gradient * local_gradient

    return gradient
