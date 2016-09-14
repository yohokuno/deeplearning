class Unit:
    def __init__(self, *parents):
        self.parents = parents
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


class Variable(Unit):
    def __init__(self, value=None):
        self.value = value
        super().__init__()

    def evaluate(self):
        return self.value


class Sum(Unit):
    def evaluate(self):
        return sum(parent.evaluate() for parent in self.parents)

    def get_gradient(self, parent):
        return Variable(1)


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


def differentiate(target, variable):
    gradients = []

    for child in variable.get_children():
        local_gradient = child.get_gradient(variable)

        if child is target:
            gradients.append(local_gradient)
        else:
            child_gradient = differentiate(target, child)
            if type(child_gradient) is Product:
                child_gradient.add_parent(local_gradient)
                gradients.append(child_gradient)
            else:
                gradients.append(Product(child_gradient, local_gradient))

    if len(gradients) == 0:
        gradient = Variable(0)
    elif len(gradients) == 1:
        gradient = gradients[0]
    else:
        gradient = Sum(*gradients)
    return gradient
