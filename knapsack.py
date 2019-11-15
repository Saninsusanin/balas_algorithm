import numpy as np


class Stack:
    def __init__(self):
        self.vector = []

    def __len__(self):
        return len(self.vector)

    def push(self, element):
        self.vector.append(element)

    def pop(self):
        return self.vector.pop()

    def top(self):
        return self.vector[len(self.vector) - 1]

    def is_empty(self):
        return len(self) == 0


class TupleIndex:
    VERTEX = 0
    FIXED_VARS_NUM = 1
    SCORE = 2


def is_impossible(constraints_vec, cur_constraints_max):
    for x, y in zip(cur_constraints_max, constraints_vec):
        if x < y:
            return True
    return False


def is_feasible(cur_constraints_vec):
    for elem in cur_constraints_vec:
        if elem > 0:
            return False
    return True


def fix_parameters(cur_vertex, prev_fixed_vars_num, cur_fixed_vars_num, cur_score,
                   cur_constraints_vec, cur_constraints_max, constraints_matrix, costs,
                   initial_constraints_vec):
    cur_var = cur_vertex[cur_fixed_vars_num - 1] if cur_fixed_vars_num > 0 else 0
    # prev node is a parent of current
    if prev_fixed_vars_num < cur_fixed_vars_num:
        cur_column = np.transpose(constraints_matrix)[cur_fixed_vars_num - 1][:]

        # fix constraints max
        cur_constraints_max += [cur_var if x < 0 else (-1 if x == 0 else 0) for x in cur_column] * cur_column

        # fix constraints vec
        cur_constraints_vec -= (cur_var * cur_column)

        # fix score
        cur_score += cur_var * costs[cur_fixed_vars_num - 1]
        if cur_fixed_vars_num > 1 and cur_vertex[cur_fixed_vars_num - 2] == 0:
            cur_score -= cur_var * costs[cur_fixed_vars_num - 1]
    else:
        cur_constraints_vec = (initial_constraints_vec - [np.dot(cur_vertex[:cur_fixed_vars_num - 1],
                                                                 constraints_matrix[row][:cur_fixed_vars_num - 1])
                                                          for row in range(len(constraints_matrix))]
                               if cur_fixed_vars_num > 0 else np.copy(initial_constraints_vec))

        for row in range(len(constraints_matrix)):
            columns_num = len(constraints_matrix[row])
            tmp = np.dot(cur_vertex[:cur_fixed_vars_num], constraints_matrix[row][:cur_fixed_vars_num])
            for column in range(cur_fixed_vars_num, columns_num):
                tmp += constraints_matrix[row][column] if constraints_matrix[row][column] > 0 else 0
            cur_constraints_max[row] = tmp

        cur_score = np.dot(cur_vertex[:cur_fixed_vars_num - 1], costs[:cur_fixed_vars_num - 1])

    if cur_var == 0 and len(costs) > cur_fixed_vars_num > 0:
        cur_score += costs[cur_fixed_vars_num]

    return cur_constraints_vec, cur_constraints_max, cur_score


def balas_algorithm(vertex, cost_variables_vector, constraints_matrix, constraints_vector):
    # stack initialization
    stack = Stack()
    stack.push((vertex, 0))

    # current min initialization
    score = None

    length = len(cost_variables_vector)
    cur_constraints_vec = np.copy(constraints_vector)
    cur_constraints_max = np.zeros(len(constraints_vector), dtype=int)
    prev_fixed_vars_num = 0
    cur_score = 0

    nodes_num = 0
    while not stack.is_empty():
        current = stack.pop()
        cur_constraints_vec, cur_constraints_max, cur_score = fix_parameters(current[TupleIndex.VERTEX],
                                                                             prev_fixed_vars_num,
                                                                             current[TupleIndex.FIXED_VARS_NUM],
                                                                             cur_score, cur_constraints_vec,
                                                                             cur_constraints_max, constraints_matrix,
                                                                             cost_variables_vector,
                                                                             constraints_vector)

        prev_fixed_vars_num = current[TupleIndex.FIXED_VARS_NUM]
        # we don't need to push current point's children if current point doesn't satisfy the constraints
        if not is_impossible(constraints_vector, cur_constraints_max):
            # check feasibility of current node
            if is_feasible(cur_constraints_vec):
                if score is None or cur_score < score[TupleIndex.SCORE]:
                    # solution will not be more optimal than the current)
                    score = (current[TupleIndex.VERTEX], current[TupleIndex.FIXED_VARS_NUM], cur_score)
                    continue

            # prune branch in which score is already worse than least score
            if score is not None and cur_score >= score[TupleIndex.SCORE]:
                continue

            # left child initialization
            left_child = np.copy(current[TupleIndex.VERTEX])
            left_child[current[TupleIndex.FIXED_VARS_NUM]] = 0

            # right child initialization
            right_child = np.copy(current[TupleIndex.VERTEX])
            right_child[current[TupleIndex.FIXED_VARS_NUM]] = 1

            # push children to the stack
            stack.push((left_child, current[TupleIndex.FIXED_VARS_NUM] + 1))
            stack.push((right_child, current[TupleIndex.FIXED_VARS_NUM] + 1))

    print(nodes_num)
    return score[TupleIndex.VERTEX]


def test1():
    propositional_variables_vector = np.array([0, 0, 0, 0, 0, 0])
    # 1 0 1 0 1 0
    cost_variables_vector = np.array([3, 5, 6, 9, 10, 10])

    constraints_matrix = np.array([[-2, 6, -3, 4, 1, -2],
                          [-5, -3, 1, 3, -2, 1],
                          [5, -1, 4, -2, 2, -1]])

    constraints_vector = np.array([2, -2, 3])
    result = balas_algorithm(propositional_variables_vector, cost_variables_vector, constraints_matrix,
                             constraints_vector)

    print(result)


def test():
    propositional_variables_v = np.array([0, 0, 0, 0, 0, 0])
    cost_variables_vector = np.array([1, 4, 15, 19, 27, 31])

    constraints_matrix = np.array([[0, 0, -1, 0, 0, -1],
                                   [-1, 2, 0, -3, -1, 0],
                                   [-1, -1, 0, 0, 0, 1],
                                   [-2, 1, 0, 0, -1, 1],
                                   [1, -1, 0, 1, 0, 1],
                                   [1, 10, 14, 12, 18, 22]])

    constraints_vector = np.array([-1, -2, -1, 0, 0, 42])

    result = balas_algorithm(propositional_variables_v, cost_variables_vector,
                             constraints_matrix, constraints_vector)
    print([1 - x for x in result])


def test2():
    propositional_variables_v = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cost_variables_vector = np.array([2, 3, 10, 12, 15, 16, 20, 21, 30, 40])

    constraints_matrix = np.array([[0, 0, 1, 0, 0, 0, 0, -1, 0, 0],
                                   [0, 0, 1, 0, 0, 0, -1, 0, 0, 0],
                                   [0, 0, 1, 0, -1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, -1, 0, -1, -1, -1],
                                   [-1, -1, -1, -1, -1, 0, -1, 0, 0, 0],
                                   [3, 5, 8, 17, 40, 5, 10, 4, 40, 50]])

    constraints_vector = np.array([0, 0, 0, -3, -5, 82])

    result = balas_algorithm(propositional_variables_v, cost_variables_vector,
                             constraints_matrix, constraints_vector)
    print([1 - x for x in result])


def scalar_product(first_vector, second_vector):
    result = 0
    length = len(first_vector)
    for i in range(length):
        result += first_vector[i] * second_vector[i]
    return result


def get_length_of_number(number):
    length = 0
    while number >> length: length += 1
    return length


def get_bit_array(number, alignment):
    length = get_length_of_number(number)
    bit_array = [0 for _ in range(alignment - length)]
    for i in range(length - 1, -1, -1):
        bit_array.append(int((number & (1 << i)) != 0))
    return bit_array


def _is_feasible(vertex, constraints_matrix, constraints_vector):
    result = True
    length = len(constraints_vector)
    for i in range(length):
        result &= scalar_product(vertex, constraints_matrix[i]) >= constraints_vector[i]
    return result


def brute_force(cost_variables_vector, constraints_matrix, constraints_vector):
    length = len(cost_variables_vector)
    bound = 1 << length
    score = None

    for i in range(bound):
        bit_array = get_bit_array(i, length)
        if _is_feasible(bit_array, constraints_matrix, constraints_vector):
            current_score = scalar_product(bit_array, cost_variables_vector)
            if score is None or score[TupleIndex.SCORE] > current_score:
                score = (bit_array, 0, current_score)

    return score if score is None else score[TupleIndex.VERTEX]


def check():
    cost_variables_vector = [2, 3, 10, 12, 15, 16, 20, 21, 30, 40]

    constraints_matrix = [[0, 0, 1, 0, 0, 0, 0, -1, 0, 0],
                          [0, 0, 1, 0, 0, 0, -1, 0, 0, 0],
                          [0, 0, 1, 0, -1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, -1, 0, -1, -1, -1],
                          [-1, -1, -1, -1, -1, 0, -1, 0, 0, 0],
                          [3, 5, 8, 17, 40, 5, 10, 4, 40, 50]]

    constraints_vector = [0, 0, 0, -3, -5, 82]

    brute_solution = brute_force(cost_variables_vector, constraints_matrix, constraints_vector)
    print([1 - x for x in brute_solution])


test2()
check()

