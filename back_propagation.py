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

    def __eq__(self, other):
        return type(self) is type(other) and self.parents == other.parents

    def __hash__(self):
        return self.parents.__hash__()


class Constant(Unit):
    def get_gradient(self):
        return []


class Sum(Unit):
    def get_gradient(self):
        return self.parents


def back_propagation(output, targets):
    grad_table = dict()
    grad_table[output] = 1

    for variable in targets:
        build_grad(variable, output, grad_table)

    return grad_table


def build_grad(variable, output, grad_table):
    print(variable, output, grad_table)
    if variable in grad_table:
        return grad_table[variable]

    gradients = []
    for i, child in enumerate(variable.get_children()):
        d = build_grad(child, output, grad_table)
        gradient = child.get_gradient()[i]
        gradients.append(gradient)

    gradients = Sum(*gradients)
    grad_table[variable] = gradients
#    insert_nodes(gradient, output)
    print('grad_table', grad_table)
    return gradients
