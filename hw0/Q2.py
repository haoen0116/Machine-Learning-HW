import PIL
import os,sys,math
from PIL import Image
im = Image.open(sys.argv[1])
rgb_im = im.convert('RGB')
[xSize,ySize] = im.size
for i in range(xSize):
    for j in range(ySize):
        [r, g, b] = rgb_im.getpixel((i, j))  #math.floor(a)
        [r, g, b] = [math.floor(r/2), math.floor(g/2), math.floor(b/2)]
        rgb_im.putpixel((i,j),(r,g,b))
rgb_im.save("./Q2.png", 'png')
