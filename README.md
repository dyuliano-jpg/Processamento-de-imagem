# Processamento-de-imagem

import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

import cv2 # openCV
from skimage.morphology import disk # scikit-image

from os import listdir, path, makedirs, chdir # costuma ter no pc, logo nao precisa importar
from os.path import isfile, join # costuma ter no pc, logo nao precisa importar

from datetime import datetime

# lê o caminho para as fotos
path_exp1 = "C:/Users/anton/Documents/Mestrado/Experimento 1"
path_exp2 = "C:/Users/anton/Documents/Mestrado/Experimento 2" # pode tirar essa linha, basta comentar ela colocando # no inicio
paths = [path_exp1, path_exp2] # pode apagar a variavel path_exp2 desde que nao exista ela


# função que busca cada dia (pasta) do experimento
def busca_pastas_dias(caminhos):
    dia = []
    for i in range(len(caminhos)):
        dia.append([f for f in listdir(caminhos[i])])
    return dia


    # for loop para buscar o caminho da foto
path2pics = []
for i in range(len(dias)):
    for pastas in dias[i]:
        pics_paths = paths[i] + '/' + pastas
        path2pics.append([pics_paths + '/' + f for f in listdir(pics_paths)])


        # precisa criar uma pasta onde será salvo cada imagem
# Caso só tenha um dia de experimento, pode excluir o segundo caminho
new_paths = ["C:/Users/anton/Documents/Mestrado/saveexp1","C:/Users/anton/Documents/Mestrado/saveexp2"]

# loop for para criar as pastas de cada dia de experimento
for i in range(len(new_paths)):
    for pastas in dias[i]:
        if not path.exists(new_paths[i] + '/' + pastas + '_modified/'):
            makedirs(new_paths[i] + '/' + pastas + '_modified/') # a função makedirs cria as novas pastas


            def salva_img_modified(name, img, path):
    chdir(path)
    cv2.imwrite(name,img)


    # Aqui é onde a mágica acontece
def k_means(img):

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # converte a imagem do BGR para CieLAB
    L = img_lab[:, :, 0]
    a = img_lab[:, :, 1]
    b = img_lab[:, :, 2]

    pixel_vals = b.flatten() # transforma o canal b em um vetor
    pixel_vals = np.float32(pixel_vals) # converte os números para 'float32'

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # critérios para o kmeans do OpenCV

    # Since we are interested in only actual leaf pixels, we choose 2 clusters
    # one cluster for actual leaf pixels and other for unwanted background pixels.

    K = 2 # número de clusters
    retval, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS) # aplica o Kmeans do OpenCV
    centers = np.uint8(centers)

    pixel_labels = labels.reshape(img_lab.shape[0], img_lab.shape[1])

    (unique, counts) = np.unique(pixel_labels, return_counts=True)
    frequencies = np.asarray((unique, counts)).T

    # cria mascara de pixels
    if frequencies[0,1] > frequencies[1,1]:
        mask = np.where(pixel_labels, 1, 0).astype(np.uint8)
    else:
        mask = np.where(pixel_labels, 0, 1).astype(np.uint8)

    # Aplicação de transformações morfológicas
    #kernel_open = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(10), iterations = 1) # consultar a função morphologyEx caso duvida

    final_img = cv2.bitwise_and(img, img, mask = opening) # junta as imagem com a mascara para resultar em img final

    return final_img


    final_start=datetime.now()
for i in range(2): # caso seja necessário, alterar o valor dentro do 'range' se der o erro 'index out of range'
    for j in range(len(path2pics[i][:])):
        caminho = path2pics[i][j]

        img_name = listdir(paths[0] + '/' + dias[0][i])[j]

        img = cv2.imread(caminho, 1)

        img_preprocessed = k_means(img)

        salva_img_modified(img_name, img_preprocessed, new_paths[0] + '/' + dias[0][i] + '_modified')

final_end=datetime.now()

print('tempo de execução do pré-processamento: ', final_end-final_start)


final_start=datetime.now()
for i in range(3): # caso seja necessário, alterar o valor dentro do 'range' se der o erro 'index out of range'
    for j in range(len(path2pics[i+3][:])):
        caminho = path2pics[i+3][j]

        img_name = listdir(paths[1] + '/' + dias[1][i])[j]

        img = cv2.imread(caminho, 1)

        img_preprocessed = k_means(img)

        salva_img_modified(img_name, img_preprocessed, new_paths[1] + '/' + dias[1][i] + '_modified')

final_end=datetime.now()

print('tempo de execução do pré-processamento: ', final_end-final_start)





    
