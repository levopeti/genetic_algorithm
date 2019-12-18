import cv2
import numpy as np
import json


def triangle_draw_ff(image, phenotypes, imshow=False):
    # heigth = image.shape[0]  # x
    # width = image.shape[1]  # y

    result_image = np.zeros_like(image)

    for phenotype in phenotypes:
        current_image = np.zeros_like(image)
        x1, y1, x2, y2, x3, y3, r, g, b, alpha = phenotype

        # draw a triangle
        vertices = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
        pts = vertices.reshape((-1, 1, 2))
        cv2.polylines(current_image, [pts], isClosed=True, color=(r, g, b), thickness=1)

        # fill it
        cv2.fillPoly(current_image, [pts], color=(r, g, b))

        # transparency
        result_image[current_image > 0] = ((1. - alpha) * result_image + alpha * current_image)[current_image > 0]

    fitness_value = np.sum((result_image - image) * (result_image - image)) / np.product(image.shape)

    if imshow:
        cv2.imshow("img", result_image)
        cv2.waitKey(0)

    return fitness_value


def circle_draw_ff(image, phenotypes, imshow=False):
    # heigth = image.shape[0]  # x
    # width = image.shape[1]  # y

    result_image = np.zeros_like(image)

    for phenotype in phenotypes:
        current_image = np.zeros_like(image)
        x, y, radius, r, g, b, alpha = phenotype

        # draw filled circle
        current_image = cv2.circle(current_image, (int(x), int(y)), int(radius), (r, g, b), thickness=-1)

        # transparency
        result_image[current_image > 0] = ((1. - alpha) * result_image + alpha * current_image)[current_image > 0]

    fitness_value = np.sum((result_image - image) * (result_image - image)) / np.product(image.shape)

    if imshow:
        cv2.imshow("img", result_image)
        cv2.waitKey(0)

    return fitness_value


if __name__ == "__main__":
    path = "/home/AD.ADASWORKS.COM/levente.peto/projects/research/genetic_algorithm/logs/gen-cd-19-12-16-16:04/result.txt"
    with open(path) as json_file:
        data = json.load(json_file)

    phenotypes = np.array(data["real_global_best_genes"]).reshape(-1, 7)
    image = cv2.imread("/home/AD.ADASWORKS.COM/levente.peto/projects/research/kislevi.jpg")
    # triangle_draw_ff(image, phenotypes, imshow=True)
    circle_draw_ff(image, phenotypes, imshow=True)

