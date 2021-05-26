import numpy as np


def mean_color_matrix(image):
    rows = len(image)
    cols = len(image[0])
    feature_matrix = np.zeros((rows, cols))
    for y in range(len(image)):
        for x in range(len(image[y])):
            color_vector = image[y][x]
            feature = int((int(color_vector[0]) + int(color_vector[1]) + int(color_vector[2])) / 3)
            feature_matrix[y][x] = feature

    return feature_matrix


def convert_to_one_d(matrix):
    one_d_vector = np.array([])
    for row in matrix:
        one_d_vector = np.concatenate((one_d_vector, row), axis=None)

    return one_d_vector


if __name__ == '__main__':
    my_image = [[[1, 2, 3],    [4, 5, 6],    [6, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]

    my_feature_color_matrix = mean_color_matrix(my_image)
    feature_vector = convert_to_one_d(my_feature_color_matrix)
    print(feature_vector)
