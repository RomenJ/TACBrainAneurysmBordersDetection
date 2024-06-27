import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import os

def process_image(image_path):
    try:
        # Lee la imagen JPG
        im = imageio.imread(image_path)
    except FileNotFoundError:
        print(f"Archivo no encontrado: {image_path}")
        return None, None
    except Exception as e:
        print(f"Error al leer la imagen {image_path}: {e}")
        return None, None

    print(f"Procesando imagen: {image_path}")
    print("Tipo devuelto:", type(im))
    print('Metadata de la imagen:')

    # Comprueba si 'im' tiene el atributo 'meta' y se imprime si existe.
    if hasattr(im, 'meta'):
        print(im.meta)
        print("Claves o número de metadatos disponible:")
        print(im.meta.keys())
    else:
        print("No hay metadatos disponibles para esta imagen.")

    print("Contenido del archivo:")
    print(im)
    print('Forma del array de imagen:', im.shape)

    # Convertir a escala de grises
    im_gray = np.dot(im[...,:3], [0.2989, 0.5870, 0.1140])

    # Definición del kernel para la detección de bordes verticales
    weights = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    # Convolución de la imagen en escala de grises con el filtro de detección de bordes
    edges = ndi.convolve(im_gray, weights)

    return edges, im

def calculate_histogram(image):
    hist, bin_edges = np.histogram(image.flatten(), bins=256, range=(0, 255))
    return hist

# Lista de imágenes a procesar
image_files = ['35.jpg', '36.jpg', '37.jpg', '38.jpg']
output_files = ['imageAnalisys/borderdetectrionImg_35.jpg',
                'imageAnalisys/borderdetectrionImg_36.jpg',
                'imageAnalisys/borderdetectrionImg_37.jpg',
                'imageAnalisys/borderdetectrionImg_38.jpg']

# Crear directorio de salida si no existe
os.makedirs('imageAnalisys', exist_ok=True)

# Acumular histogramas de todas las imágenes
total_hist = np.zeros(256)

# Procesar cada imagen
for image_file, output_file in zip(image_files, output_files):
    edges, im = process_image(image_file)
    if edges is None or im is None:
        continue
    
    hist = calculate_histogram(im)
    total_hist += hist

    # Dibujar la imagen original y la imagen con bordes
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(im)
    ax[0].axis('off')
    ax[0].set_title('Original')
    
    ax[1].imshow(edges, cmap='seismic', vmin=-150, vmax=150)
    ax[1].axis('off')
    ax[1].set_title('Detección de bordes')

    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.show()

# Mostrar el histograma acumulado
plt.figure()
plt.plot(total_hist)
plt.title("Histograma acumulado")
plt.xlabel("Valor de píxel")
plt.ylabel("Frecuencia")
plt.savefig('imageAnalisys/total_histogram.jpg')
plt.show()
