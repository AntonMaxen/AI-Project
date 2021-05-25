import math


def softmax(input_vector):
    e = math.e
    scaled_vector = [e**num for num in input_vector]
    softmax_vector = [num / sum(scaled_vector) for num in scaled_vector]
    return softmax_vector


def relu(input_vector):
    return [max(0, num) for num in input_vector]


def main():
    input_vector = [1, 2, 3, 4, 1, 2, 3]
    softmax_vector = softmax(input_vector)
    print(softmax_vector)
    relu_vector = relu(input_vector)
    print(relu_vector)



if __name__ == '__main__':
    main()
