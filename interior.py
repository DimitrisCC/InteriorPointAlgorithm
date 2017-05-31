import numpy as np

# function to minimize
def f(x, e):
    x1 = x[0,0]
    x2 = x[1,0]
    return x1**2 + x2**2 - e*(np.log(6-2*x1-3*x2) + np.log(-1-2*x1+x2))

# Gradient of f
def grad(x, e):
    x1 = x[0,0]
    x2 = x[1,0]
    grad = [2*x1 + e*(2./(-2*x1+x2-1) + 2./(-2*x1-3*x2+6)), 2*x2 + e*(1./(2*x1-x2+1) - 3./(-2*x1-3*x2+6))]
    return np.array(grad).reshape((2,1))

# Hessian Matrix of f
def hessian(x, e):
    x1 = x[0,0]
    x2 = x[1,0]
    H = [[2 + e*(4./((2*x1-x2+1)**2) + 4./(2*x1+3*x2-6)**2), e*(-2./((2*x1-x2+1)**2) + 6./(2*x1+3*x2-6)**2)], \
         [e*(-2./((2*x1-x2+1)**2) + 6./(2*x1+3*x2-6)**2), 2 + e*(1./((2*x1-x2+1)**2) + 9./(2*x1+3*x2-6)**2)]]
    return np.array(H)

def backtracking(x0, delta, e, c):
    t = 1
    while True:
        x1 = x0 + t*delta
        if f(x1,e) <= f(x0,e) + c*grad(x0,e).T @ (x1 - x0):
            break
        else:
            t /= 2
    return x1
    
def newton(x0, e, c, tol=1e-3):
    assert(c > 0 and c < 1)
    while True:
        delta = - np.linalg.inv(hessian(x0, e)) @ grad(x0, e)
        x1 = backtracking(x0, delta, e, c=c)
        if np.linalg.norm(x1 - x0) < tol:
            break
        else:
            x0 = x1
    return x0

def interior_point(x0, e, c, tol=1e-5):
    assert(tol > 0)
    while True:
        xe = newton(x0, e, c=c)
        if e <= tol:
            break
        else:
            x0 = xe
            e /= 2
    return x0

xe = interior_point(x0=np.array([-1, 1]).reshape((2,1)), e=1, c=0.5)
print("Minimum at:", xe.flatten())
print("Minimum:", f(xe,1))