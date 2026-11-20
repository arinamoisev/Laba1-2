import numpy as np
from PIL import Image, ImageTk
from math import *
import tkinter as tk
from numpy.random import normal
from numba import njit

# 7 задание. Барицентрические координаты
@njit(cache=True)
def bar(x,y,x0,y0,x1,y1,x2,y2):
     lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) /((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
     lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) /((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
     lambda2 = 1.0 - lambda0 - lambda1
     return lambda0, lambda1, lambda2

# 8 задание. Отрисовка треугольника
@njit(cache=True)
def triangle(img_mat, zbuf,px0,py0,z0,px1,py1,z1,px2,py2,z2, color):
    xmin = floor(min(min(px0, px1), px2))
    xmax = ceil(max(max(px0, px1), px2))
    ymin = floor(min(min(py0, py1), py2))
    ymax = ceil(max(max(py0, py1), py2))
    if (ymin < 0): ymin = 0
    if (xmin < 0): xmin = 0
    if (xmax > img_mat.shape[1]): xmax = img_mat.shape[1]-1
    if (ymax > img_mat.shape[0]): ymax = img_mat.shape[0]-1
    for j in range(ymin, ymax+1):
        for i in range(xmin, xmax+1):
            t = bar(i+0.5, j+0.5, px0,py0,px1,py1,px2,py2)
            lambda0, lambda1, lambda2 = t
            if lambda0>=0 and lambda1>=0 and lambda2>=0:
                z = z0*lambda0+z1*lambda1+z2*lambda2
                if z<zbuf[j,i]:
                    img_mat[j, i, 0] = color[0]
                    img_mat[j, i, 1] = color[1]
                    img_mat[j, i, 2] = color[2]
                    zbuf[j, i] = z

# задание 11. Нормали
@njit(cache=True)
def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    ax = x1 - x2;  ay = y1 - y2;  az = z1 - z2
    bx = x1 - x0;  by = y1 - y0;  bz = z1 - z0
    nx = ay*bz - az*by
    ny = az*bx - ax*bz
    nz = ax*by - ay*bx
    return nx, ny, nz



class Scene:
    def __init__(self, filename):
        file = open('model.obj')
        self.v = [] # будут храниться точки
        self.f = []
        self.light = [0.0,0.0,1.0]

        for s in file:
            sp = s.split()
            if sp[0] == 'v':
                self.v.append([float(x) for x in sp[1:4]])
            if sp[0] == 'f':
                self.f.append([int(sp[1].split('/')[0]), int(sp[2].split('/')[0]), int(sp[3].split('/')[0])])
        self.v = np.array(self.v)
        self.f = np.array(self.f)
        self.light = np.array(self.light)

@njit(cache=True)
def render_model(img_mat, zbuf, v, f, light, scale, cx, cy,color):
    H, W = img_mat.shape[0:2]
    # рисуем зайца треугольниками 10-14 задание
    for i in range(len(f)):  # f[i] номера вершин i треугольника v[f[i][0]-1][0] полигон вершина координата
        f_0, f_1, f_2 = f[i][0] - 1, f[i][1] - 1, f[i][2] - 1

        x0, y0, z0 = v[f_0][0], v[f_0][1], v[f_0][2]
        x1, y1, z1 = v[f_1][0], v[f_1][1], v[f_1][2]
        x2, y2, z2 = v[f_2][0], v[f_2][1], v[f_2][2]

        px0, py0 = x0 * scale/z0 + cx, -y0 * scale/z0 + cy
        px1, py1 = x1 * scale/z1 + cx, -y1 * scale/z1 + cy
        px2, py2 = x2 * scale/z2 + cx, -y2 * scale/z2 + cy

        nx, ny, nz = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        dot = nx * light[0] + ny * light[1] + nz * light[2]
        length_n = sqrt(nx * nx + ny * ny + nz * nz)
        length_light = sqrt(light[0] * light[0] + light[1] * light[1] + light[2] * light[2])
        cos_a = dot / (length_n * length_light)
        if cos_a >= 0: continue
        r = int(color[0] * -cos_a)
        g = int(color[1] * -cos_a)
        b = int(color[2] * -cos_a)
        triangle(img_mat, zbuf, px0, py0, z0, px1, py1, z1, px2, py2, z2, (r, g, b))

@njit(cache=True)
def render_model2(img_mat, zbuf, v, f, light, scale, cx, cy,color):
    H, W = img_mat.shape[0:2]
    # рисуем зайца треугольниками 10-14 задание
    for i in range(len(f)):  # f[i] номера вершин i треугольника v[f[i][0]-1][0] полигон вершина координата
        f_0, f_1, f_2 = f[i][0] - 1, f[i][1] - 1, f[i][2] - 1

        x0, y0, z0 = v[f_0][0], v[f_0][1], v[f_0][2]
        x1, y1, z1 = v[f_1][0], v[f_1][1], v[f_1][2]
        x2, y2, z2 = v[f_2][0], v[f_2][1], v[f_2][2]

        px0, py0 = x0 * scale + cx, -y0 * scale + cy
        px1, py1 = x1 * scale + cx, -y1 * scale + cy
        px2, py2 = x2 * scale + cx, -y2 * scale + cy

        nx, ny, nz = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        dot = nx * light[0] + ny * light[1] + nz * light[2]
        length_n = sqrt(nx * nx + ny * ny + nz * nz)
        length_light = sqrt(light[0] * light[0] + light[1] * light[1] + light[2] * light[2])
        cos_a = dot / (length_n * length_light)
        if cos_a >= 0: continue
        r = int(color[0] * -cos_a)
        g = int(color[1] * -cos_a)
        b = int(color[2] * -cos_a)
        triangle(img_mat, zbuf, px0, py0, z0, px1, py1, z1, px2, py2, z2, (r, g, b))

class Renderer:
    def __init__(self, W,H,scale):
        self.color = [255, 0, 0]  # по умолчанию красный
        self.flag = 1
        self.H = H
        self.W = W
        self.scale = scale
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0

        self.alpha = 0
        self.beta = 0
        self.gamma = 0
        self.tX = 0.0
        self.tY = -0.03
        self.tZ = 1

        self.img_mat = np.zeros((H, W, 3), dtype=np.uint8)  # матрица
        self.zbuf = np.full((H, W), np.inf, dtype=np.float64)  # z буфер

    def render(self, scene):
        self.img_mat[...]=0
        self.zbuf[...]=np.inf

        cosAlpha = cos(self.alpha)
        cosBeta = cos(self.beta)
        cosGamma = cos(self.gamma)

        sinAlpha = sin(self.alpha)
        sinBeta = sin(self.beta)
        sinGamma = sin(self.gamma)

        Rx = np.array([[1,0,0],[0,cosAlpha,sinAlpha],[0,-sinAlpha,cosAlpha]])
        Ry = np.array([[cosBeta,0,sinBeta],[0,1,0],[-sinBeta,0,cosBeta]])
        Rz = np.array([[cosGamma,sinGamma,0],[-sinGamma,cosGamma,0],[0,0,1]])
        R = Rx@Ry@Rz
        v_r = scene.v@R.T + [self.tX, self.tY, self.tZ]
        if self.flag==1:
            render_model(self.img_mat, self.zbuf, v_r, scene.f, scene.light, self.scale, self.cx, self.cy, self.color)
        else:
            render_model2(self.img_mat, self.zbuf, v_r, scene.f, scene.light, self.scale, self.cx, self.cy, self.color)

        
class App:
    def __init__(self, scene, renderer, win_w = 900, win_h = 900):
        self.scene = scene
        self.renderer = renderer
        # Окно верхнего уровня
        self.root = tk.Tk()
        # Размеры окна
        self.root.geometry(f"{win_w}x{win_h}")
        # Область для рисования внутри окна
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.focus_set()
        # Обновление изображения при изменении размеров окна
        self.canvas.bind('<Configure>', lambda e: self._refit())
        # Привязка обработчика клавиш
        self.root.bind('<Key>', self.on_key)
        # Изображение внутри окна
        self.photo = None
        self.full_image = None
        self.render()
        self.make_blue()
    def make_red(self):
        self.renderer.color[0]=0
        self.renderer.color[2] = 255
        self.render()
        self.root.after(500,self.make_blue)

    def make_blue(self):
        self.renderer.color[0]=255
        self.renderer.color[2] = 0
        self.render()
        self.root.after(500,self.make_red)

    def on_key(self, event):
        k = event.keysym
        if k == 'plus':
            self.renderer.scale *= 1.1
        elif k == 'minus':
            self.renderer.scale /= 1.1
        elif k=='1':
            self.renderer.alpha+=0.1
        elif k=='2':
            self.renderer.beta += 0.1
        elif k=='3':
            self.renderer.gamma += 0.1
        elif k=='4':
            self.renderer.tX+=0.01
        elif k=='5':
            self.renderer.tX-=0.01
        elif k=='6':
            self.renderer.tY+=0.01
        elif k=='7':
            self.renderer.tY-=0.01
        elif k=='8':
            self.renderer.tZ+=0.01
        elif k=='9':
            self.renderer.tZ-=0.01
        elif k=='q':
            self.renderer.flag = 1
        elif k=='w':
            self.renderer.flag = 0
        elif k == 'space':
            Image.fromarray(self.renderer.img_mat).save('img.png')
            return 'break'
        else:
            return
        self.render()
        # Не обрабатываем дальше – return 'brake'
        return 'break'

    def render(self):
        self.renderer.render(self.scene)
        self.full_image = Image.fromarray(self.renderer.img_mat)
        self._refit()

    def _refit(self):
        # Если ещё не отрендерилось, пропускаем
        if self.full_image is None: return
        # Ширина и высота области для рисования
        ww = self.canvas.winfo_width()
        wh = self.canvas.winfo_height()
        # Если вдруг сделали область слишком маленькой
        if ww < 10 or wh < 10: return
        # Вписать с сохранением пропорций
        s = min(ww / self.renderer.W, wh / self.renderer.H)
        dw = max(1, int(self.renderer.W * s))
        dh = max(1, int(self.renderer.H * s))
        disp = self.full_image.resize((dw, dh), Image.BILINEAR)
        # Очень важно сохранить ImageTk.PhotoImage(disp) в поле класса
        # Если оно будет локальным, то удалится после выхода сборщиком мусора
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.delete('all')
        self.canvas.create_image(ww // 2, wh // 2, image=self.photo)

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    scene = Scene("model.obj")
    renderer = Renderer(1000, 1000, 5000.0)
    App(scene, renderer, 500, 500).run()