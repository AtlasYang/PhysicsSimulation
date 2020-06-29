''''
2020년 6월 25일
창체동아리 ANN 혹은 물리학2 세부능력특기사항 작성용 단진동 시뮬레이션
python과 pygame 이용하여 시각화

*동아리 부원들과 학습할 용도이므로...
최대한 패키지 사용을 지양할 것
코드를 직관적이고 간결하게 작성할 것
물리학의 교육과정 및 미적분 교과의 내용을 이용할 것
'''

import pygame
from pygame.locals import *
import numpy as np
import time

FPS = 10000      #초당 프레임
WIDTH = 800
HEIGHT = 600
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

dt = 0.01  #시간의 순간변화량. 작을 수록 정밀
a = np.array([1.5, 234.7])
#pygame 좌표 대입용 성분 정수 변환
def vector_int(v):
    t = []
    for comp in v:
        t.append(int(comp))
    return t

#2차원 유클리드 벡터 y좌표 반전(pygame 카르테시안 좌표계 보정)
def y_c(v):
    return (v[0], (-1)*v[1])

#위치벡터 종점간 거리
def distance(pos1, pos2):
    dim = len(pos1)
    t = 0
    for i in range(dim):
        t += ((pos1[i] - pos2[i]) ** 2)
    return np.sqrt(t)

def L2norm(v):
    t = 0
    for i in v:
        t += i**2
    return np.sqrt(t)

'''
질점은 클래스로 구현.
질점의 위치, 속도, 가속도 및 힘은 모두 질점의 위치를
원점으로 하는 2차원 위치벡터 순서쌍으로 나타낸다.
'''
class particle():
    def __init__(self, mass=1, pos=np.array([0, 0]), velocity = np.array([0, 0])):
        self.m = mass
        self.x = pos
        self.v = velocity
        self.a = np.array([0, 0])
        self.dx = np.array([0, 0])
        self.dv = np.array([0, 0])
        self.net_f = 0

    def move(self, f):
        self.net_f = f
        self.a = self.net_f / self.m  #a = F / m
        #print('가속도: '+ str(self.a))

        self.dv = self.a * dt
        print('델타속도: '+ str(self.dv))

        self.v = self.v + self.dv
        print('속도: '+ str(self.v))

        self.dx = self.v * dt
        print('위치 변화량: '+ str(self.dx))

        #print('이전 위치: '+ str(self.x))
        self.x = self.x + self.dx
        #print('나중 위치: '+ str(self.x))

        print('')



'''
단진동 시스템 구현
fixed_pos: 고정점
obj: 운동하는 질점
length: 실의 길이 => 초기 질점과 고정점으로부터 자동 계산
G = 중력 계수
'''


def main(init_pos = np.array([250, 400]), gravity_constant = 9.8 * 10, obj_m = 1):
    #pygame 그래픽 초기화
    global FPSCLOCK, DISPLAYSURF
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT))
    DISPLAYSURF.fill(WHITE)

    #조작 변인 정의
    G = gravity_constant

    fixed_pos = np.array([400, 150])
    p1 = np.array([400, 300])
    baselen = distance(p1, fixed_pos)

    obj_initial_pos = init_pos
    obj_mass = obj_m = 1
    obj = particle(obj_mass, obj_initial_pos)

    length = distance(fixed_pos, obj_initial_pos)

    temp1 = np.dot((p1 - fixed_pos), (obj.x - fixed_pos)) / (length * baselen)
    theta_max = np.arccos(temp1)

    length_var = length

    latest_time = 0
    period = 0.0
    ttime = 0
    period_list = []
    while True:
        #모든 이벤트 무시
        for event in pygame.event.get():
            break

        p1 = (400, 300)
        temp2 = np.dot((p1 - fixed_pos), (obj.x - fixed_pos)) / (length_var * baselen)
        #print('코사인각도: ' + str(temp2))
        if obj.x[0] < fixed_pos[0]:
            theta = np.arccos(temp2)
        else:
            theta = -1 * np.arccos(temp2)
        print('각도: ' + str(np.rad2deg(theta)))

        theta = np.abs(theta)

        gravity_force = np.array([0, obj.m * G])

        tension_dir = fixed_pos - obj.x
        tension_mag = obj.m * G * np.cos(theta) + ((obj.m * (L2norm(obj.v))**2) / length_var)
        tension_vec = tension_dir / L2norm(tension_dir) * tension_mag

        force = gravity_force + tension_vec
        #알짜힘 적용
        obj.move(force)

        #실 길이 재계산
        length_var = distance(fixed_pos, obj.x)
        #print('실 길이: '+ str(length_var))

        #surface에 그리기 위해서 정수화
        pos = vector_int(obj.x)
        #print('위치: '+str(obj.x))

        if obj.x[0] - fixed_pos[0] <= 5 and obj.x[0] - fixed_pos[0] > 0:
            if time.time() - latest_time >= 0.5:
                ttime = time.time()
                period = (ttime - latest_time) * 2
                period_list.append(period)
                latest_time = ttime

        #물체, 고정점, 실 그리기
        DISPLAYSURF.fill(WHITE)
        pygame.draw.aaline(DISPLAYSURF, RED, fixed_pos, pos, 2)
        pygame.draw.circle(DISPLAYSURF, BLACK, fixed_pos, 5, 0)
        pygame.draw.circle(DISPLAYSURF, BLUE, pos, 15, 0)

        #힘 그리기
        t_pos = vector_int(pos + tension_vec)
        g_pos = vector_int(pos + gravity_force)
        f_pos = vector_int(pos + obj.net_f)
        pygame.draw.aaline(DISPLAYSURF, BLACK, pos, f_pos, 2)
        pygame.draw.aaline(DISPLAYSURF, BLACK, pos, g_pos, 2)
        pygame.draw.aaline(DISPLAYSURF, BLACK, pos, t_pos, 2)

        pygame.draw.circle(DISPLAYSURF, BLACK, f_pos, 4, 0)
        pygame.draw.circle(DISPLAYSURF, BLACK, g_pos, 4, 0)
        pygame.draw.circle(DISPLAYSURF, BLACK, t_pos, 4, 0)

        #가속도, 속도 막대그래프
        k = 240 / L2norm(obj.v)
        k1 = 200 / L2norm(obj.a)

        pygame.draw.rect(DISPLAYSURF, RED, (600, 270 - k1 * np.abs(obj.a[0]), 30, k1 * np.abs(obj.a[0])))
        pygame.draw.rect(DISPLAYSURF, RED, (660, 270 - k1 * np.abs(obj.a[1]), 30, k1 * np.abs(obj.a[1])))
        pygame.draw.rect(DISPLAYSURF, RED, (720, 270 - k1 * L2norm(obj.a), 30, k1 * L2norm(obj.a)))

        pygame.draw.rect(DISPLAYSURF, GREEN, (600, 570 - np.abs(obj.v[0]), 30, np.abs(obj.v[0])))
        pygame.draw.rect(DISPLAYSURF, GREEN, (660, 570 - np.abs(obj.v[1]), 30, np.abs(obj.v[1])))
        pygame.draw.rect(DISPLAYSURF, GREEN, (720, 570 - L2norm(obj.v), 30, L2norm(obj.v)))

        font = pygame.font.Font('freesansbold.ttf',20)

        text_ax = font.render('A_x', True, RED)
        text_ay = font.render('A_y', True, RED)
        text_a = font.render('A', True, RED)

        text_vx = font.render('V_x', True, GREEN)
        text_vy = font.render('V_y', True, GREEN)
        text_v = font.render('V', True, GREEN)

        DISPLAYSURF.blit(text_ax,(600,275))
        DISPLAYSURF.blit(text_ay,(660,275))
        DISPLAYSURF.blit(text_a,(720,275))

        DISPLAYSURF.blit(text_vx,(600,575))
        DISPLAYSURF.blit(text_vy,(660,575))
        DISPLAYSURF.blit(text_v,(720,575))

        #주기 및 반지름 표시
        text = font.render('Period: ' + str(np.round_(period, 5)) + 's', True, BLACK)
        DISPLAYSURF.blit(text,(20,20))
        text2 = font.render('Radius: ' + str(np.round_(length_var, 5)) + 'pixel', True, BLACK)
        DISPLAYSURF.blit(text2,(20,50))
        text3 = font.render('Speed: ' + str(np.round_(L2norm(obj.v), 5)) + 'pixel/s', True, BLACK)
        DISPLAYSURF.blit(text3,(20,90))
        #text4 = font.render('Period Var: ' + str(np.round_(np.var(period_list), 5)) + 's^2', True, BLACK)
        #DISPLAYSURF.blit(text4,(20,120))

        pygame.display.set_caption('Simple Pendulum Simulation')
        pygame.display.update()
        FPSCLOCK.tick(FPS)

if __name__ == '__main__':
    print('단진자 시뮬레이션 by ANN')
    g = float(input('중력상수[9.8*10]: '))
    m = float(input('물체 질량[1]: '))
    r = float(input('초기x좌표(0~400)[300]: '))
    p = float(input('초기y좌표(150~600)[300]: '))
    if g == 0:
        g = 9.8 * 10
    if m == 0:
        m = 1
    if r == 0:
        r = 300
    if p == 0:
        p = 300
    main(np.array([r, p]), g, m)
