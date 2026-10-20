import numpy as np
from PIL import Image
from math import *
from numpy.random import normal

img_mat = np.zeros((1000,1000,3), dtype = np.uint8)  # матрица
zbuf = np.full((1000,1000), np.inf, dtype = np.float64) # z буфер

# рисуем квадрат красивый, если хотим градиент, то функцию, а не цвета
#for i in range(1000):
#      for j in range(1000):
#         img_mat[i,j] = 255, 200, 180 # градация серого, полутон (i+j)%256 и ноль после j

# рисуем линии
def line(img_mat,x0,y0,x1,y1):
    d_max = max(abs(floor(x0) - floor(x1)), abs(floor(y0) - floor(y1)))
    L = d_max + 1
    if L == 1:
        px = floor(x0)
        py = floor(y0)
        img_mat[py, px] = 255
        return

    Dx = (x1 - x0) / (L - 1)
    Dy = (y1 - y0) / (L - 1)
    x, y = x0, y0
    for _ in range(L):  # рисуем линию
        img_mat[floor(y), floor(x)] = 255, 140, 185
        x += Dx
        y += Dy
#line(img_mat,100, 200, 334, 344)
#line(img_mat,109, 334, 654, 234)

file = open('model.obj')
v = []  # храниться будуь точки
f = []
for s in file:
    sp = s.split()
    if sp[0]=='v':
        v.append([float(x) for x in sp[1:4]])
    if sp[0]=='f':
        f.append([int(sp[1].split('/')[0]),int(sp[2].split('/')[0]),int(sp[3].split('/')[0])])
#print(v)
#print(f)



# рисуем зайца по точкам
for i in range(len(v)):
    x = 5000*v[i][0]+500
    y = -5000*v[i][1]+500  # выравниывем по центру, добавляем половину
 #   img_mat[floor(y), floor(x)] = 255, 140, 185  # красный зеленый синий

 # 7 задание. Барицентрические координаты
def bar(x,y,x0,y0,x1,y1,x2,y2):
     denom = (x0-x2)*(y1-y2)-(x1-x2)*(y0-y2)
     if (denom==0):
         return
     lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) /((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
     lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) /((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
     lambda2 = 1.0 - lambda0 - lambda1
     return lambda0, lambda1, lambda2

# 8 задание. Отрисовка треугольника
def triangle(img_mat, zbuf,x0,y0,z0,x1,y1,z1,x2,y2,z2, color):
    xmin = floor(min(x0, x1, x2))
    xmax = ceil(max(x0, x1, x2))
    ymin = floor(min(y0, y1, y2))
    ymax = ceil(max(y0, y1, y2))
    if (ymin < 0): ymin = 0
    if (xmin < 0): xmin = 0
    if (xmax > 1000): xmax = 1000
    if (ymax > 1000): ymax = 1000
    #color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
    for j in range(ymin, ymax+1):
        for i in range(xmin, xmax+1):
            t = bar(i+0.5, j+0.5, x0,y0,x1,y1,x2,y2)
            lambda0, lambda1, lambda2 = t
            if lambda0>=0 and lambda1>=0 and lambda2>=0:
                z = z0*lambda0+z1*lambda1+z2*lambda2
                if z<zbuf[j,i]:
                    img_mat[j, i] = color
                    zbuf[j, i] = z

#triangle(img_mat, 100,100,400,150,250,500)

# задание 11. Нормали
def normal(p0,p1,p2):
    a = [p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]]
    b = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]]
    nx = a[1]*b[2]-a[2]*b[1]
    ny = a[2]*b[0]-a[0]*b[2]
    nz = a[0]*b[1]-a[1]*b[0]
    return [nx, ny, nz]

# рисуем зайца треугольниками 10-14 задание
for i in range(len(f)):  # f[i] номера вершин i треугольника v[f[i][0]-1][0] полигон вершина координата
    x0 = 5000*v[f[i][0]-1][0]+500
    y0 = -5000*v[f[i][0]-1][1]+500
    z0 = 5000*v[f[i][0]-1][2]+500

    x1 = 5000 * v[f[i][1] - 1][0] + 500
    y1 = -5000 * v[f[i][1] - 1][1] + 500
    z1 = 5000 * v[f[i][1] - 1][2] + 500

    x2 = 5000 * v[f[i][2] - 1][0] + 500
    y2 = -5000 * v[f[i][2] - 1][1] + 500
    z2 = 5000 * v[f[i][2] - 1][2] + 500

    n = normal(v[f[i][0]-1], v[f[i][1]-1], v[f[i][2]-1])
    l = [0.0,0.0,1.0]
    cos_a = np.dot(n,l)/(np.linalg.norm(n)*np.linalg.norm(l))
    if cos_a >= 0: continue
    triangle(img_mat, zbuf, x0, y0, z0, x1, y1, z1, x2, y2, z2,(int(-255.0 * cos_a), 0, 0))


img = Image.fromarray(img_mat)  # сохранение
img.save('img.png')
