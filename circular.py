'''
2020년 6월 25일
창체동아리 ANN 혹은 물리학2 세부능력특기사항 작성용 물리학 시뮬레이션
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

FPS = 1000      #초당 프레임
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
        print('가속도: '+ str(self.a))

        self.dv = self.a * dt
        #print('델타속도: '+ str(self.dv))

        self.v = self.v + self.dv
        print('속도: '+ str(self.v))
        print('속력: '+ str(L2norm(self.v)))

        self.dx = self.v * dt
        print('위치 변화량: '+ str(self.dx))

        #print('이전 위치: '+ str(self.x))
        self.x = self.x + self.dx
        #print('나중 위치: '+ str(self.x))

        print('')



'''
등속 원운동 시스템 구현
Center = 고정 중심
Radius = 회전 반지름
G = 중력 계수
'''


def main(gravity_constant = 9.8*100, mass = 100, radius = 150):
    #pygame 그래픽 초기화
    global FPSCLOCK, DISPLAYSURF
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT))
    DISPLAYSURF.fill(WHITE)

    #조작 변인 정의
    G = gravity_constant

    Center = np.array([400, 300])
    Mass = mass
    Radius = radius

    vtemp_x = np.sqrt(G * Mass / Radius)

    obj_initial_pos = np.array([400, 300 - Radius])
    obj_mass = 1
    obj_initial_velocity = np.array([vtemp_x, 0])
    obj = particle(obj_mass, obj_initial_pos, obj_initial_velocity)

    radius_var = distance(Center, obj_initial_pos)

    latest_time = 0
    period = 0.0
    ttime = 0
    period_list = []
    r_list = []
    while True:
        #모든 이벤트 무시
        for event in pygame.event.get():
            break

        mag = (G * Mass * obj.m) / (radius_var ** 2)
        print('힘 크기: ' + str(mag))

        dir_vec = Center - obj.x
        dir_vec = dir_vec / L2norm(dir_vec)
        dir_vec = dir_vec * mag

        f_x = dir_vec[0]
        f_y = dir_vec[1]

        force = np.array([f_x, f_y])

        #알짜힘 적용
        obj.move(force)

        #실 길이 재계산
        radius_var = distance(Center, obj.x)
        print('반지름: '+ str(radius_var))
        r_list.append(radius_var)

        #surface에 그리기 위해서 정수화
        pos = vector_int(obj.x)
        print('위치: '+str(obj.x))


        if obj.x[0] - Center[0] <= 5 and obj.x[0] - Center[0] > 0 and obj.x[1] < Center[1]:
            if time.time() - latest_time >= 0.5:
                ttime = time.time()
                period = (ttime - latest_time)
                period_list.append(period)
                latest_time = ttime


        #물체, 고정점, 실 그리기
        DISPLAYSURF.fill(WHITE)
        pygame.draw.aaline(DISPLAYSURF, RED, Center, pos, 2)
        pygame.draw.circle(DISPLAYSURF, BLACK, Center, 3, 0)
        pygame.draw.circle(DISPLAYSURF, BLUE, pos, 15, 0)

        #가속도, 속도 막대그래프
        k = 240 / L2norm(obj.v)
        k1 = 200 / L2norm(obj.a)

        pygame.draw.rect(DISPLAYSURF, RED, (600, 270 - k1 * np.abs(obj.a[0]), 30, k1 * np.abs(obj.a[0])))
        pygame.draw.rect(DISPLAYSURF, RED, (660, 270 - k1 * np.abs(obj.a[1]), 30, k1 * np.abs(obj.a[1])))
        pygame.draw.rect(DISPLAYSURF, RED, (720, 270 - k1 * L2norm(obj.a), 30, k1 * L2norm(obj.a)))

        pygame.draw.rect(DISPLAYSURF, GREEN, (600, 570 - k * np.abs(obj.v[0]), 30, k * np.abs(obj.v[0])))
        pygame.draw.rect(DISPLAYSURF, GREEN, (660, 570 - k * np.abs(obj.v[1]), 30, k * np.abs(obj.v[1])))
        pygame.draw.rect(DISPLAYSURF, GREEN, (720, 570 - k * L2norm(obj.v), 30, k * L2norm(obj.v)))

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
        text2 = font.render('Radius: ' + str(np.round_(radius_var, 5)) + 'pixel', True, BLACK)
        DISPLAYSURF.blit(text2,(20,50))
        text3 = font.render('Speed: ' + str(np.round_(6.18 * radius_var / period, 5)) + 'pixel/s', True, BLACK)
        DISPLAYSURF.blit(text3,(20,90))
        text4 = font.render('Radius Var: ' + str(np.round_(np.var(r_list), 5)) + 'pixel^2', True, BLACK)
        DISPLAYSURF.blit(text4,(20,120))



        pygame.display.set_caption('Uniform Circular Motion Simulation')
        pygame.display.update()
        FPSCLOCK.tick(FPS)


if __name__ == '__main__':
    print('등속 원운동 시뮬레이션 by ANN')
    g = float(input('중력상수[1000]: '))
    m = float(input('중심 질량[100]: '))
    r = float(input('반지름(0~300)[150]: '))
    if g == 0:
        g = 1000
    if m == 0:
        m = 100
    if r == 0:
        r = 150
    main(g, m, r)
