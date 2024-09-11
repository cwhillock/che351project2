import numpy as np 
import math
from tabulate import tabulate
import matplotlib.pyplot as mpl

tank1A = 0.5 * math.pi * math.pow(1,2)
tank2A = 2 * 1.5
pipe1A = 0.5 * math.pi * math.pow(0.1,2) 
pipe1H = 0.5
pipe2A = 0.5 * math.pi * math.pow(0.1,2) 
Qt = 0.05
h1=0
h2=0
g = 9.81
qp1 = 0
qp2 = 0
'system variables/initial conditions'

def dh1(qt,qp1):
    return (qt - qp1)/tank1A

def dh2(qp1,qp2):
    return (qp1-qp2)/tank2A

def qp1v1(h1,h2):
    'valid when h2 > H and h1 > h2'
    return pipe1A * math.pow(2*g*(h1-h2),0.5)

def qp1v2(h1):
    'valid when h1 > H and h2 < H'
    return pipe1A * math.pow(2*g*(h1-pipe1H),0.5)

def qp1v3(h1,h2):
    'valid when h1 > H and h2 > h1'
    return -1 * pipe1A * math.pow(2*g*(h2-h1),0.5)

def qp1v4(h2):
    'valid when h2 > H and h1 < H'
    return -1 * pipe1A * math.pow(2*g*(h2-pipe1H),0.5)

def qp2v1(h2):
    'valid always'
    return pipe2A * math.pow(2*g*h2,0.5)

def setqp1(h1,h2):
    if (h1 > pipe1H) and (h2 < pipe1H):
        return qp1v2(h1)
    if (h2 > pipe1H) and (h1 > h2):
        return qp1v1(h1,h2)
    if (h1>pipe1H) and (h2>h1):
        return qp1v3(h1,h2)
    if (h1<pipe1H) and (h2>pipe1H):
        return qp1v4(h2)
    return 0
        
numrow = 500
dt = 1
currentt = 0
'initial conditions for rk4'

header = np.array(["t","Q(t)","h1","h2","Qp1","Qp2"])
'header array'

values = np.empty([numrow,6])
values[0,0] = currentt
values[0,1] = Qt
values[0,2] = h1
values[0,3] = h2
values[0,4] = qp1
values[0,5] = qp2
'values array'

for i in range(1, numrow):
    k1 = dt * dh1(Qt,setqp1(h1,h2))
    z1 = dt * dh2(setqp1(h1,h2),qp2v1(h2))
    k2 = dt * dh1(Qt,setqp1(h1 + 0.5 * k1,h2 + 0.5 * z1))
    z2 = dt * dh2(setqp1(h1 + 0.5 * k1,h2 + 0.5 * z1),qp2v1(h2 + 0.5 * z1))
    k3 = dt * dh1(Qt,setqp1(h1 + 0.5 * k2,h2 + 0.5 * z2))
    z3 = dt * dh2(setqp1(h1 + 0.5 * k2,h2 + 0.5 * z2),qp2v1(h2 + 0.5 * z2))
    k4 = dt * dh1(Qt,setqp1(h1 + k3,h2 + z3))
    z4 = dt * dh2(setqp1(h1 + k3,h2 + z3),qp2v1(h2 + z3))
    'estimating system of ODEs with runge kutta 4'

    h1 = h1 + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
    h2 = h2 + (1/6) * (z1 + 2 * z2 + 2 * z3 + z4)
    if h2 < 0:
        h2 = 0
    currentt = currentt + dt
    qp1 = setqp1(h1,h2)
    qp2 = qp2v1(h2)
    'setting new values'
    values[i,0] = currentt
    values[i,1] = Qt
    values[i,4] = qp1
    values[i,5] = qp2
    values[i,2] = h1
    values[i,3] = h2
    'writing to array'

combined = np.vstack((header,values))
print(tabulate(combined))
'combining arrays'

mpl.plot(values[:,0],values[:,2],values[:,0],values[:,3])
mpl.show()
'plotting'