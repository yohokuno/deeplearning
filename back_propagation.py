class Unit:
    def __init__(self, *parents):
        self.parents = parents
        for parent in parents:
            parent.add_child(self)

        self.children = []

    def get_parents(self):
        return self.parents

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children


class Constant(Unit):
    def __init__(self, value=None):
        self.value = value
        super(Constant, self).__init__()

    def evaluate(self):
        return self.value


class Sum(Unit):
    def get_gradient(self):
        return [Constant(1) for _ in self.parents]

    def evaluate(self):
        return sum(parent.evaluate() for parent in self.parents)


class Product(Unit):
    def get_gradient(self):
        if len(self.parents) == 1:
            return [Constant(1.0)]
        elif len(self.parents) == 2:
            return self.parents[1], self.parents[0]
        return [Product(*(self.parents[:i] + self.parents[i+1:])) for i in range(len(self.parents))]

    def evaluate(self):
        result = 1.0
        for parent in self.parents:
            result *= parent.evaluate()
        return result


def back_propagation(output, targets):
    grad_table = dict()
    grad_table[output] = 1

    for variable in targets:
        build_grad(variable, output, grad_table)

    return grad_table


def build_grad(variable):
    gradients = []
    for child in variable.get_children():
        for parent, gradient in zip(child.get_parents(), child.get_gradient()):
            if parent is variable:
                # found gradient respect to parent
                break
        gradients.append(gradient)

    if len(gradients) == 1:
        gradient = gradients[0]
    else:
        gradient = Sum(*gradients)
    return gradient
