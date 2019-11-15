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
        return self.__len__() == 0


class TupleIndex:
    VERTEX = 0
    FIXED_ARGS_NUM = 1
    SCORE = 2


# utils
def scalar_product(first_vector, second_vector):
    result = 0
    length = len(first_vector)

    for i in range(length): result += first_vector[i] * second_vector[i]

    return result


def is_impossible(vertex, number_of_fixed_variables, constraints_matrix, constraints_vector):
    result = False
    number_of_rows = len(constraints_vector)

    for row in range(number_of_rows):
        fixed_variables_contribution = scalar_product(vertex[:number_of_fixed_variables],
                                                      constraints_matrix[row][:number_of_fixed_variables])

        if constraints_vector[row] > fixed_variables_contribution:
            maximum = 0
            number_of_columns = len(vertex)

            for column in range(number_of_fixed_variables, number_of_columns):
                maximum += constraints_matrix[row][column] if constraints_matrix[row][column] > 0 else 0

            result |= maximum < (constraints_vector[row] - fixed_variables_contribution)

    return result


def is_feasible(vertex, constraints_matrix, constraints_vector):
    result = True
    length = len(constraints_vector)

    for i in range(length): result &= scalar_product(vertex, constraints_matrix[i]) >= constraints_vector[i]

    return result


# for 1-node counts scalar product, for 0-node look-ahead and count score of sibling 1-node(if exists)
def score_function(vertex, number_of_fixed_variables, cost_variables_vector):
    score = scalar_product(vertex, cost_variables_vector)

    if vertex[number_of_fixed_variables - 1] == 0 and number_of_fixed_variables < len(vertex):
        score += cost_variables_vector[number_of_fixed_variables]

    return score


def balah_algorithm(vertex, cost_variables_vector, constraints_matrix, constraints_vector):
    # initializing of the stack
    stack = Stack()
    stack.push((vertex, 0))

    # initializing of the current max
    score = None

    # this is only need to not recalculate on every iteration number of boolean variables
    length = len(cost_variables_vector)

    while not stack.is_empty():
        current = stack.pop()

        # we don't need to push "siblings" of current point if current point isn't satisfy the constraints
        if not is_impossible(current[TupleIndex.VERTEX], current[TupleIndex.FIXED_ARGS_NUM], constraints_matrix,
                             constraints_vector) and current[TupleIndex.FIXED_ARGS_NUM] < length:
            # initializig score of current vertex(to get rid of excess calculations)
            current_score = score_function(current[TupleIndex.VERTEX], current[TupleIndex.FIXED_ARGS_NUM],
                                           cost_variables_vector)

            # this step checks feasibility of current node. It is necessary for future updates of the score
            if is_feasible(current[TupleIndex.VERTEX], constraints_matrix, constraints_vector):

                # this step initialize or update score
                if score is None or current_score < score[TupleIndex.SCORE]:
                    score = (current[TupleIndex.VERTEX], current[TupleIndex.FIXED_ARGS_NUM], current_score)

                    # it is skip adding of the siblings(further, the solution will not be more optimal than the current)
                    continue

            # prune branch in which score is already worse than least score
            if score is not None and current_score >= score[TupleIndex.SCORE]:
                continue

            # initializing of the left sibling
            left_sibling = current[TupleIndex.VERTEX][:]
            left_sibling[current[TupleIndex.FIXED_ARGS_NUM]] = 0

            # initializing of the right sibling
            right_sibling = current[TupleIndex.VERTEX][:]
            right_sibling[current[TupleIndex.FIXED_ARGS_NUM]] = 1

            # push siblings to the stack
            stack.push((left_sibling, current[TupleIndex.FIXED_ARGS_NUM] + 1))
            stack.push((right_sibling, current[TupleIndex.FIXED_ARGS_NUM] + 1))

    return score[TupleIndex.VERTEX]

# book example(search minimum)
def test_1():
    propositional_variables_v = [0, 0, 0, 0, 0, 0]

    cost_variables_vector = [3, 5, 6, 9, 10, 10]

    constraints_matrix = [[-2, 6, -3, 4, 1, -2],
                          [-5, -3, 1, 3, -2, 1],
                          [5, -1, 4, -2, 2, -1]]

    constraints_vector = [2, -2, 3]

    result = balah_algorithm(propositional_variables_v, cost_variables_vector, constraints_matrix, constraints_vector)
    print(result)

# search maximum
def test_2():
    propositional_variables_v = [0, 0, 0, 0, 0, 0]
    cost_variables_vector = [31, 15, 1, 27, 4, 19]
    constraints_matrix = [[-1, -1, 0, 0, 0, 0],
                          [0, 0, -1, -1, -1, -1],
                          [1, 0, -1, 0, -1, 0],
                          [1, 0, -1, 0, 1, 0],
                          [1, 0, 1, 0, -1, 0],
                          [18, 14, 3, 22, 18, 12]]
    constraints_vector = [-1, -2, -1, 0, 0, 42]

    result = balah_algorithm(propositional_variables_v, cost_variables_vector, constraints_matrix, constraints_vector)
    print([1 - element for element in result])

test_2()