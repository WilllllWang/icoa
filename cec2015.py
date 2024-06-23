import numpy as np
import math


def main(): 
    x = [2,2,2,2,2,2,2,2,2,2]
    y = [1,2,3,4,5,6,7,8,9,10]
    z = [1.0,-2.9,3.8,-4.7,5.6,-6.5,7.4,-8.3,9.2,-10.1]
    r = 5
    x *= 2
    print(cf7(z))







# ----------Basic functions----------
# f1
def high_conditioned_elliptic_function(x):
    coef = 1e6
    d = len(x)
    return np.sum([coef**((i-1)/(d-1))*(x[i]**2) for i in range(d)])


# f2
def cigar_function(x):
    coef = 1e6
    d = len(x)
    return np.sum([coef*(x[i]**2) for i in range(1, d)]) + x[0]**2


# f3
def discus_function(x):
    coef = 1e6
    d = len(x)
    return np.sum([(x[i]**2) for i in range(1, d)]) + coef*(x[0]**2)


# f4 
def rosenbrock_function_standard(x, y): # helper function
    coef = 100
    return coef*((y-(x**2))**2) + ((1-x)**2)

def rosenbrock_function(x): 
    d = len(x)
    return np.sum([rosenbrock_function_standard(x[i], x[i+1]) for i in range(d-1)])


# f5
def ackley_function(x):
    d = len(x)
    first = np.sum([(x[i]**2) for i in range(d)])
    second = np.sum([math.cos(2*math.pi*x[i]) for i in range(d)])
    return (-20)*math.exp((-0.2)*math.sqrt(first/d)) - math.exp(second/d) + 20 + math.exp(1)


# f6
def weierstrass_function(x):
    d = len(x)
    a = 0.5
    b = 3
    kmax = 20
    first = 0
    second = 0
    for k in range(kmax+1):
        second += d * ((a**k) * math.cos(2 * math.pi * (b**k) * 0.5))
    for i in range(d):
        for k in range(kmax+1):
            first += (a**k) * math.cos(2 * math.pi * (b**k) * (x[i] + 0.5))
    return (first - second)

        
# f7     
def griewank_function(x):
    coef = 1/4000
    d = len(x)
    first = np.sum([(x[i]**2) for i in range(d)]) * coef
    second = np.prod([math.cos(x[i] / math.sqrt(i+1)) for i in range(d)]) 
    return first - second + 1


# f8
def rastrigin_function(x):
    d = len(x)
    coef = 10
    return np.sum([(x[i]**2) - (coef*math.cos(2*math.pi*x[i])) + coef for i in range(d)])


# f9
def modified_schwefel_function(x):
    d = len(x)
    res = 0
    for i in range(d):
        z = x[i] + 420.9687462275036
        if abs(z) < 500:
            res += z * math.sin((abs(z)**0.5))
        elif z > 500:
            res += (500-(z%500)) * math.sin(math.sqrt(abs((500-(z%500))))) - ((z-500)**2) / (10000*d)
        else:
            res += ((abs(z)%500)-500) * math.sin(math.sqrt(abs(((abs(z)%500)-500)))) - ((z+500)**2) / (10000*d)
    return ((418.9829 * d) - res)


# f10 
def katsuura_function(x):
    d = len(x)
    power = 10 / (d**1.2)
    res = 1
    for i in range(d):
        eq = 0
        for j in range(1, 33):
            eq += (abs((2**j)*x[i] - round((2**j)*x[i]))) / (2**j)
        res *= (1 + i*eq)**power
    return (10/(d**2)*res) - 10/(d**2)


# f11
def happycat_function(x):
    d = len(x)
    first = np.sum([(x[i]) for i in range(d)])
    second = np.sum([(x[i]**2) for i in range(d)])
    return (abs(second - d))**0.25 + ((0.5*second + first) / d) + 0.5


# f12
def hgbat_function(x):
    d = len(x)
    first = np.sum([(x[i]) for i in range(d)])
    second = np.sum([(x[i]**2) for i in range(d)])
    return (abs((second**2) - (first**2)))**0.5 + ((0.5*second + first) / d) + 0.5


# f13
def expanded_griewank_plus_rosenbrock_function(x):
    d = len(x)
    arr = np.zeros(d)
    arr[d-1] = rosenbrock_function_standard(x[d-1], x[0])
    for i in range(d-1):
        arr[i] = rosenbrock_function_standard(x[i], x[i+1])
    return griewank_function(arr)
    

# f14
def scaffer_fsix_function(x, y): # helper function
    return 0.5 + ((math.sin(math.sqrt(x**2 + y**2)))**2 - 0.5) / ((1+0.001*(x**2 + y**2))**2)

def expanded_scaffer_fsix_function(x):
    d = len(x)
    arr = np.zeros(d)
    arr[d-1] = scaffer_fsix_function(x[d-1], x[0])
    for i in range(d-1):
        arr[i] = scaffer_fsix_function(x[i], x[i+1])
    return np.sum(arr)



# ----------Main Functions----------
# Unimodal Functions
# F1 rotated high conditioned elliptic function
def uf1(x):
    sr_x = shift_and_rotate_function(x, 1, 0, 1, 1, 1.0)
    return high_conditioned_elliptic_function(sr_x) + 100         # f1


# F2 rotated cigar function
def uf2(x):
    sr_x = shift_and_rotate_function(x, 2, 0, 1, 1, 1.0)
    return cigar_function(sr_x) + 200                             # f2



# Multimodal Functions
# F3 shifted and rotated ackley function
def mf1(x):
    sr_x = shift_and_rotate_function(x, 3, 0, 1, 1, 1.0)
    return ackley_function(sr_x) + 300                            # f5


# F4 shifted and rotated rastrigin function
def mf2(x):
    shift_rate = 5.12 / 100
    sr_x = shift_and_rotate_function(x, 4, 0, 1, 1, shift_rate)
    return rastrigin_function(sr_x) + 400                         # f8


# F5 shifted and rotated schwefel function
def mf3(x):
    shift_rate = 1000 / 100
    sr_x = shift_and_rotate_function(x, 5, 0, 1, 1, shift_rate)
    return modified_schwefel_function(sr_x) + 500                 # f9



# ----------Hybrid function----------
# F6
def hf1(x):
    d = len(x)
    sr_x = shift_and_rotate_function(x, 6, 0, 1, 1, 1.0)
    s = get_shuffled(6, d)
    y = np.zeros(d)
    for i in range(d):
        y[i] = sr_x[s[i]-1] # sr_x in random order
    p = np.array([0.3, 0.3, 0.4])
    N = 3
    p *= d
    g = [0] * N
    g[0] = 0
    for i in range(1, N):
        g[i] = int(g[i-1] + p[i-1]) # Slice array into N parts
    
    return modified_schwefel_function(x[ :g[1]]) + rastrigin_function(x[g[1]: g[2]]) + high_conditioned_elliptic_function(x[g[2]: ]) + 600


# F7
def hf2(x):
    d = len(x)
    sr_x = shift_and_rotate_function(x, 7, 0, 1, 1, 1.0)
    s = get_shuffled(7, d)
    y = np.zeros(d)
    for i in range(d):
        y[i] = sr_x[s[i]-1] # sr_x in random order
    p = np.array([0.2, 0.2, 0.3, 0.3])
    N = 4
    p *= d
    g = [0] * N
    g[0] = 0
    for i in range(1, N):
        g[i] = int(g[i-1] + p[i-1]) # Slice array into N parts
    
    return griewank_function(x[ :g[1]]) + weierstrass_function(x[g[1]: g[2]]) + rosenbrock_function(x[g[2]: g[3]]) + expanded_scaffer_fsix_function(x[g[3]: ]) + 700


# F8
def hf3(x):
    d = len(x)
    sr_x = shift_and_rotate_function(x, 8, 0, 1, 1, 1.0)
    s = get_shuffled(8, d)
    y = np.zeros(d)
    for i in range(d):
        y[i] = sr_x[s[i]-1] # sr_x in random order
    p = np.array([0.1, 0.2, 0.2, 0.2, 0.3])
    N = 5
    p *= d
    g = [0] * N
    g[0] = 0
    for i in range(1, N):
        g[i] = int(g[i-1] + p[i-1]) # Slice array into N parts
    
    return expanded_scaffer_fsix_function(x[ :g[1]]) + hgbat_function(x[g[1]: g[2]]) + rosenbrock_function(x[g[2]: g[3]]) + modified_schwefel_function(x[g[3]: g[4]]) + high_conditioned_elliptic_function(x[g[4]: ]) + 800



# ----------Composition functions----------
# F9
def cf1(x):
    d = len(x)
    N = 3
    i = 0  
    s_x1 = shift_and_rotate_function(x, 9, i*d, 1, 0, 1000.0/100.0)
    i = 1
    s_x2 = shift_and_rotate_function(x, 9, i*d, 1, 0, 5.12/100.0)
    sr_x2 = rotate_function(s_x2, 9, i*(d**2))
    i = 2
    s_x3 = shift_and_rotate_function(x, 9, i*d, 1, 0, 5.0/100.0)
    sr_x3 = rotate_function(s_x3, 9, i*(d**2))

    fit = [0] * N
    fit[0] = modified_schwefel_function(s_x1)
    fit[1] = rastrigin_function(sr_x2)
    fit[2] = hgbat_function(sr_x3)
    sigma = [20, 20, 20]
    bias = [0, 100, 200]
    return cf_cal(x, 9, fit, N, bias, sigma) + 900


# F10
def cf2(x):
    d = len(x)
    N = 3
    i = 0  
    s_x1 = shift_and_rotate_function(x, 10, i*d, 1, 0, 1.0)
    sr_x1 = rotate_function(s_x1, 10, i*(d**2))
    i = 1
    s_x2 = shift_and_rotate_function(x, 10, i*d, 1, 0, 1.0)
    sr_x2 = rotate_function(s_x2, 10, i*(d**2))
    i = 2
    s_x3 = shift_and_rotate_function(x, 10, i*d, 1, 0, 1.0)
    sr_x3 = rotate_function(s_x3, 10, i*(d**2))

    fit = [0] * N
    fit[0] = hf1(sr_x1)
    fit[1] = hf2(sr_x2)
    fit[2] = hf3(sr_x3)
    sigma = [10, 30, 50]
    bias = [0, 100, 200]
    return cf_cal(x, 10, fit, N, bias, sigma) + 1000


# F11
def cf3(x):
    d = len(x)
    N = 5
    i = 0  
    s_x1 = shift_and_rotate_function(x, 11, i*d, 1, 0, 5.0 / 100.0)
    sr_x1 = rotate_function(s_x1, 11, i*(d**2))
    i = 1
    s_x2 = shift_and_rotate_function(x, 11, i*d, 1, 0, 5.12 / 100.0)
    sr_x2 = rotate_function(s_x2, 11, i*(d**2))
    i = 2
    s_x3 = shift_and_rotate_function(x, 11, i*d, 1, 0, 1000.0 / 100.0)
    sr_x3 = rotate_function(s_x3, 11, i*(d**2))
    i = 3
    s_x4 = shift_and_rotate_function(x, 11, i*d, 1, 0, 0.5 / 100.0)
    sr_x4 = rotate_function(s_x4, 11, i*(d**2))
    i = 4
    s_x5 = shift_and_rotate_function(x, 11, i*d, 1, 0, 1.0)
    sr_x5 = rotate_function(s_x5, 11, i*(d**2))

    fit = [0] * N
    fit[0] = hgbat_function(sr_x1)
    fit[0] *= 10000.0 / 1000.0
    fit[1] = rastrigin_function(sr_x2)
    fit[1] *= 10000.0 / 1e3
    fit[2] = modified_schwefel_function(sr_x3)
    fit[2] *= 10000.0 / 4e3
    fit[3] = weierstrass_function(sr_x4)
    fit[3] *= 10000.0 / 400.0
    fit[4] = high_conditioned_elliptic_function(sr_x5)
    fit[4] *= 10000.0 / 1e10
    sigma = [10, 10, 10, 20, 20]
    bias = [0, 100, 200, 300, 400]
    return cf_cal(x, 11, fit, N, bias, sigma) + 1100


# F12
def cf4(x):
    d = len(x)
    N = 5
    i = 0  
    s_x1 = shift_and_rotate_function(x, 12, i*d, 1, 0, 1000.0 / 100.0)
    sr_x1 = rotate_function(s_x1, 12, i*(d**2))
    i = 1
    s_x2 = shift_and_rotate_function(x, 12, i*d, 1, 0, 5.12 / 100.0)
    sr_x2 = rotate_function(s_x2, 12, i*(d**2))
    i = 2
    s_x3 = shift_and_rotate_function(x, 12, i*d, 1, 0, 1.0)
    sr_x3 = rotate_function(s_x3, 12, i*(d**2))
    i = 3
    s_x4 = shift_and_rotate_function(x, 12, i*d, 1, 0, 1.0)
    sr_x4 = rotate_function(s_x4, 12, i*(d**2))
    i = 4
    s_x5 = shift_and_rotate_function(x, 12, i*d, 1, 0, 5.0 / 100.0)
    sr_x5 = rotate_function(s_x5, 12, i*(d**2))

    fit = [0] * N
    fit[0] = modified_schwefel_function(sr_x1)
    fit[0] *= 10000.0 / 4e3
    fit[1] = rastrigin_function(sr_x2)
    fit[1] *= 10000.0 / 1e3
    fit[2] = high_conditioned_elliptic_function(sr_x3)
    fit[2] *= 10000.0 / 1e10
    fit[3] = expanded_scaffer_fsix_function(sr_x4)
    fit[3] *= 10000.0 / 1000.0
    fit[4] = happycat_function(sr_x5)
    fit[4] *= 10000.0 / 1e3
    sigma = [10, 20, 20, 30, 30]
    bias = [0, 100, 100, 200, 200]
    return cf_cal(x, 12, fit, N, bias, sigma) + 1200


# F13
def cf5(x):
    d = len(x)
    N = 5
    i = 0  
    s_x1 = shift_and_rotate_function(x, 13, i*d, 1, 0, 1.0)
    sr_x1 = rotate_function(s_x1, 13, i*(d**2))
    i = 1
    s_x2 = shift_and_rotate_function(x, 13, i*d, 1, 0, 5.12 / 100.0)
    sr_x2 = rotate_function(s_x2, 13, i*(d**2))
    i = 2
    s_x3 = shift_and_rotate_function(x, 13, i*d, 1, 0, 1.0)
    sr_x3 = rotate_function(s_x3, 13, i*(d**2))
    i = 3
    s_x4 = shift_and_rotate_function(x, 13, i*d, 1, 0, 1000.0 / 100.0)
    sr_x4 = rotate_function(s_x4, 13, i*(d**2))
    i = 4
    s_x5 = shift_and_rotate_function(x, 13, i*d, 1, 0, 1.0)
    sr_x5 = rotate_function(s_x5, 13, i*(d**2))

    fit = [0] * N
    fit[0] = hf3(sr_x1)
    fit[1] = rastrigin_function(sr_x2)
    fit[1] *= 10000.0 / 1e3
    fit[2] = hf1(sr_x3)
    fit[3] = modified_schwefel_function(sr_x4)
    fit[3] *= 10000.0 / 4e3
    fit[4] = expanded_scaffer_fsix_function(sr_x5)
    fit[4] *= 10000.0 / 1000.0
    sigma = [10, 10, 10, 20, 20]
    bias = [0, 100, 200, 300, 400]
    return cf_cal(x, 13, fit, N, bias, sigma) + 1300


# F14
def cf6(x):
    d = len(x)
    N = 7
    i = 0  
    s_x1 = shift_and_rotate_function(x, 14, i*d, 1, 0, 5.0 / 100.0)
    sr_x1 = rotate_function(s_x1, 14, i*(d**2))
    i = 1
    s_x2 = shift_and_rotate_function(x, 14, i*d, 1, 0, 5.0 / 100.0)
    sr_x2 = rotate_function(s_x2, 14, i*(d**2))
    i = 2
    s_x3 = shift_and_rotate_function(x, 14, i*d, 1, 0, 1000.0 / 100.0)
    sr_x3 = rotate_function(s_x3, 14, i*(d**2))
    i = 3
    s_x4 = shift_and_rotate_function(x, 14, i*d, 1, 0, 1.0)
    sr_x4 = rotate_function(s_x4, 14, i*(d**2))
    i = 4
    s_x5 = shift_and_rotate_function(x, 14, i*d, 1, 0, 1.0)
    sr_x5 = rotate_function(s_x5, 14, i*(d**2))
    i = 5
    s_x6 = shift_and_rotate_function(x, 14, i*d, 1, 0, 1.0)
    sr_x6 = rotate_function(s_x6, 14, i*(d**2))
    i = 6
    s_x7 = shift_and_rotate_function(x, 14, i*d, 1, 0, 5.12 / 100.0)
    sr_x7 = rotate_function(s_x7, 14, i*(d**2))

    fit = [0] * N
    fit[0] = happycat_function(sr_x1)
    fit[0] *= 10000.0 / 1e3
    fit[1] = expanded_griewank_plus_rosenbrock_function(sr_x2)
    fit[1] *= 10000.0 / 4e3
    fit[2] = modified_schwefel_function(sr_x3)
    fit[2] *= 10000.0 / 4e3
    fit[3] = expanded_scaffer_fsix_function(sr_x4)
    fit[3] *= 10000.0 / 1000.0
    fit[4] = high_conditioned_elliptic_function(sr_x5)
    fit[4] *= 10000.0 / 1e10
    fit[5] = cigar_function(sr_x6)
    fit[5] *= 10000.0 / 1e10 
    fit[6] = rastrigin_function(sr_x7)
    fit[6] *= 10000.0 / 1e3
    sigma = [10, 20, 30, 40, 50, 50, 50]
    bias = [0, 100, 200, 300, 300, 400, 400]
    return cf_cal(x, 14, fit, N, bias, sigma) + 1400


# F15
def cf7(x):
    d = len(x)
    N = 10
    i = 0  
    s_x1 = shift_and_rotate_function(x, 15, i*d, 1, 0, 5.12 / 100.0)
    sr_x1 = rotate_function(s_x1, 15, i*(d**2))
    i = 1
    s_x2 = shift_and_rotate_function(x, 15, i*d, 1, 0, 0.5 / 100.0)
    sr_x2 = rotate_function(s_x2, 15, i*(d**2))
    i = 2
    s_x3 = shift_and_rotate_function(x, 15, i*d, 1, 0, 5.0 / 100.0)
    sr_x3 = rotate_function(s_x3, 15, i*(d**2))
    i = 3
    s_x4 = shift_and_rotate_function(x, 15, i*d, 1, 0, 1000.0 / 100.0)
    sr_x4 = rotate_function(s_x4, 15, i*(d**2))
    i = 4
    s_x5 = shift_and_rotate_function(x, 15, i*d, 1, 0, 2.048/100.0)
    sr_x5 = rotate_function(s_x5, 15, i*(d**2))
    i = 5
    s_x6 = shift_and_rotate_function(x, 15, i*d, 1, 0, 5.0 / 100.0)
    sr_x6 = rotate_function(s_x6, 15, i*(d**2))
    i = 6
    s_x7 = shift_and_rotate_function(x, 15, i*d, 1, 0, 5.0 / 100.0)
    sr_x7 = rotate_function(s_x7, 15, i*(d**2))
    i = 7
    s_x8 = shift_and_rotate_function(x, 15, i*d, 1, 0, 1.0)
    sr_x8 = rotate_function(s_x8, 15, i*(d**2))
    i = 8
    s_x9 = shift_and_rotate_function(x, 15, i*d, 1, 0, 5.0 / 100.0)
    sr_x9 = rotate_function(s_x9, 15, i*(d**2))
    i = 9
    s_x10 = shift_and_rotate_function(x, 15, i*d, 1, 0, 1.0)
    sr_x10 = rotate_function(s_x10, 15, i*(d**2))


    fit = [0] * N
    fit[0] = rastrigin_function(sr_x1)
    fit[0] *= 100.0 / 1e3
    fit[1] = weierstrass_function(sr_x2)
    fit[1] *= 100.0 / 400.0
    fit[2] = happycat_function(sr_x3)
    fit[2] *= 100.0 / 1e3
    fit[3] = modified_schwefel_function(sr_x4)
    fit[3] *= 100.0 / 4e3
    fit[4] = rosenbrock_function(sr_x5)
    fit[4] *= 100.0 / 1e5
    fit[5] = hgbat_function(sr_x6)
    fit[5] *= 100.0 / 1000.0
    fit[6] = katsuura_function(sr_x7)
    fit[6] *= 100.0 / 1e7
    fit[7] = expanded_scaffer_fsix_function(sr_x8)
    fit[7] *= 1000.0 / 100.0
    fit[8] = expanded_griewank_plus_rosenbrock_function(sr_x9)
    fit[8] *= 100.0 / 4e3
    fit[9] = ackley_function(sr_x10)
    fit[9] *= 100.0 / 1e5
    sigma = [10, 10, 20, 20, 30, 30, 40, 40, 50, 50]
    bias = [0, 100, 100, 200, 200, 300, 300, 400, 400, 500]
    return cf_cal(x, 15, fit, N, bias, sigma) + 1500













# Shift
def get_shifted(f): 
    with open(f"input_data/shift_data_{f}.txt") as shifted_data:
        return shifted_data.read().split()

def shift_function(x, f, start):
    d = len(x)
    s_x = np.zeros(d)
    shifted = get_shifted(f)[start: start + d]
    for i in range(d):
        s_x[i] = x[i] - float(shifted[i])
    return s_x

# Rotate 
def get_rotated(f, d): 
    with open(f"input_data/M_{f}_D{d}.txt") as rotatedData:
        return rotatedData.read().split()
        
def rotate_function(x, f, start):
    d = len(x)
    r_x = np.zeros(d)
    rotated = get_rotated(f, d)[start: start + (d**2)]
    for i in range(d):
        for j in range(d):
            r_x[i] = (x[j] * float(rotated[i*d + j])) + r_x[i]
    return r_x

# Shifted and rotated
# s_x is shifted
# r_x is rotated
# sr_x is both
def shift_and_rotate_function(x, f, start, s_flag, r_flag, shift_rate):
    if s_flag and r_flag:                # s and r
        s_x = shift_function(x, f, start)
        s_x *= shift_rate
        return rotate_function(s_x, f, start)
    elif s_flag and not r_flag:          # s
        s_x = shift_function(x, f, start)
        s_x *= shift_rate
        return s_x
    else:                                # r
        return rotate_function(x, f, start) 

        
# Shuffle 
def get_shuffled(f, d):
    with open(f"input_data/shuffle_data_{f}_D{d}.txt") as shuffle_data:
        sh = shuffle_data.read().split()
        for i in range(d):
            sh[i] = int(sh[i])
        return sh



# Calculate composition functions
def cf_cal(x, f, fit, N, bias, sigma):
    d = len(x)
    shifted = get_shifted(f)
    w = [0] * N
    wmax = 0
    wsum = 0
    for i in range(N):
        fit[i] += bias[i]
        for j in range(N):
            w[i] += (x[j]-float(shifted[i*d + j]))**2
        
        if w[i] != 0:
            w[i] = ((1.0/w[i])**0.5) * math.exp(-w[i] / (2 * d * (sigma[i]**2)))
        else:
            w[i] = float('inf')

        wmax = max(wmax, w[i])
    
    for i in range(N):
        wsum += w[i]
    if wmax == 0:
        for i in range(N):
            w[i] = 1
        wsum = N
    
    res = 0
    for i in range(N):
        res += w[i] / (wsum*fit[i])
    return res







if __name__ == "__main__":
    main()







