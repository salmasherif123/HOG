import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import color, draw
import numpy as np
from scipy import ndimage

def drawHog(orientation_histogram):
    s_row = 128
    s_col = 64
    # radius = min(8, 8) // 2 - 1
    orientations_arr = np.arange(9)
    orientation_bin_midpoints = (orientations_arr + .5) / 9
    dr_arr = np.sin(orientation_bin_midpoints)
    dc_arr = np.cos(orientation_bin_midpoints)
    hog_image = np.zeros((s_row, s_col))
    for r in range(16):
        for c in range(8):
            for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                centre = tuple([r * 8 + 8 // 2,c * 8 + 8 // 2])
                rr, cc = draw.line(int(centre[0] - dc),int(centre[1] + dr),int(centre[0] + dc),int(centre[1] - dr))
                hog_image[rr, cc] += orientation_histogram[r, c, o]
    return hog_image
img = mpimg.imread('among.png')
plt.imshow(img)
plt.show()
img = resize(img, (128, 64))
plt.imshow(img)
plt.show()
img = color.rgb2gray(img)
plt.imshow(img)
plt.show()

dx = ndimage.sobel(img, 0)
dy = ndimage.sobel(img, 1)


mag = np.sqrt(dx ** 2 + dy ** 2)
phase = np.zeros((128, 64))

for i in range(128):
    for j in range(64):
        if dx[i][j] != 0:
            phase[i][j] = round(np.arctan(dy[i][j] / dx[i][j]) * 100) % 180

plt.imshow(mag)
plt.show()
plt.imshow(phase)
plt.show()
mag_8_blocks = np.zeros((16, 8, 8, 8))
phase_8_blocks = np.zeros((16, 8, 8, 8))
for i in range(16):
    for j in range(8):
        mag_8_blocks[i][j] = mag[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
        phase_8_blocks[i][j] = phase[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]

pins = np.zeros((16, 8, 9))
for i in range(16):
    p = np.zeros((8, 9))
    for j in range(8):
        pinsPerBlock = np.zeros(9)
        for m in range(8):
            for n in range(8):
                ind = phase_8_blocks[i][j][m][n] / 20
                if ind > 8:
                    ind = 8
                c = math.ceil(ind)
                f = math.floor(ind)
                if c == f:
                    pinsPerBlock[c] += mag_8_blocks[i][j][m][n]
                else:
                    r1 = abs(c * 20 - phase_8_blocks[i][j][m][n]) / 20
                    r2 = abs(f * 20 - phase_8_blocks[i][j][m][n]) / 20
                    pinsPerBlock[c] += r1 * mag_8_blocks[i][j][m][n]
                    pinsPerBlock[f] += r2 * mag_8_blocks[i][j][m][n]
        p[j] = pinsPerBlock
    pins[i] = p

hogHistorgram = drawHog(pins)
print(hogHistorgram)
plt.imshow(hogHistorgram)
plt.show()

hist_16_block = np.zeros((15, 7, 36))
for i in range(15):
    hist = np.zeros((7, 36))
    for j in range(7):
        arr = pins[i * 2 - i:(i + 1) * 2 - i, j * 2 - j:(j + 1) * 2 - j]
        arr = arr.flatten()
        sum = np.sum(arr)
        if sum != 0:
            arr /= sum
        hist[j] = arr
    hist_16_block[i] = hist

print(hist_16_block.shape)
fv = hist_16_block.flatten()
print(fv.shape)

