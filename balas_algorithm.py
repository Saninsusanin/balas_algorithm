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


def brute_force(cost_variables_vector, constraints_matrix, constraints_vector):
    length = len(cost_variables_vector)
    bound = 1 << length
    score = None

    for i in range(bound):
        bit_array = get_bit_array(i, length)

        if is_feasible(bit_array, constraints_matrix, constraints_vector):
            current_score = scalar_product(bit_array, cost_variables_vector)

            if score is None or score[TupleIndex.SCORE] > current_score:
                score = (bit_array, 0, current_score)

    return score if score is None else score[TupleIndex.VERTEX]


def vectors_is_equal(first_vector, second_vector):

    if first_vector is None:
        return True if second_vector is None else False

    result = True
    length = len(first_vector)

    for i in range(length):
        result &= (first_vector[i] == second_vector[i])

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


def balas_algorithm(cost_variables_vector, constraints_matrix, constraints_vector):
    # initializing of the stack
    stack = Stack()
    stack.push(([0 for i in range(len(cost_variables_vector))], 0))

    # initializing of the current max
    score = None

    # this is only need to not recalculate on every iteration number of boolean variables
    length = len(cost_variables_vector)

    while not stack.is_empty():
        current = stack.pop()

        # we don't need to push "siblings" of current point if current point isn't satisfy the constraints
        if not is_impossible(current[TupleIndex.VERTEX], current[TupleIndex.FIXED_ARGS_NUM], constraints_matrix,
                             constraints_vector):
            # initializing score of current vertex(to get rid of excess calculations)
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

    return score if score is None else score[TupleIndex.VERTEX]

# book example(search minimum)


def test_1():
    cost_variables_vector = [3, 5, 6, 9, 10, 10]
    constraints_matrix = [[-2, 6, -3, 4, 1, -2],
                          [-5, -3, 1, 3, -2, 1],
                          [5, -1, 4, -2, 2, -1]]
    constraints_vector = [2, -2, 3]

    balas_solution = balas_algorithm(cost_variables_vector, constraints_matrix, constraints_vector)
    brute_solution = brute_force(cost_variables_vector, constraints_matrix, constraints_vector)

    print(vectors_is_equal(balas_solution, brute_solution))

# search maximum


def test_2():
    cost_variables_vector = [1, 4, 15, 19, 27, 31]
    constraints_matrix = [[0, 0, -1, 0, 0, -1],
                          [-1, -1, 0, -1, -1, 0],
                          [-1, -1, 0, 0, 0, 1],
                          [-1, 1, 0, 0, 0, 1],
                          [1, -1, 0, 0, 0, 1],
                          [3, 18, 14, 12, 22, 18]]
    constraints_vector = [-1, -2, -1, 0, 0, 42]

    balas_solution = balas_algorithm(cost_variables_vector, constraints_matrix, constraints_vector)
    #balas_solution = balas_solution if balas_solution is None else [1 - element for element in balas_solution]
    brute_solution =  brute_force(cost_variables_vector, constraints_matrix, constraints_vector)

    print(vectors_is_equal(balas_solution, brute_solution))


def test_3():
    cost_variables_vector = [2, 3, 10, 12, 15, 16, 20, 21, 30, 40]
    constraints_matrix = [[0, 0, 1, 0, 0, 0, 0, -1, 0, 0],
                                   [0, 0, 1, 0, 0, 0, -1, 0, 0, 0],
                                   [0, 0, 1, 0, -1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, -1, 0, -1, -1, -1],
                                   [-1, -1, -1, -1, -1, 0, -1, 0, 0, 0],
                                   [3, 5, 8, 17, 40, 5, 10, 4, 40, 50]]
    constraints_vector = [0, 0, 0, -3, -2, 82]

    balas_solution = balas_algorithm(cost_variables_vector, constraints_matrix, constraints_vector)
    brute_solution = brute_force(cost_variables_vector, constraints_matrix, constraints_vector)

    print(vectors_is_equal(balas_solution, brute_solution))


test_1()
test_2()
test_3()