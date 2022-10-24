dy/dt = f(y(t), x(t))

where y is your evolving state, and x is your input signal?

If so this is straightforward. In Diffrax for example (although you could do something similar in other packages too) then you could do:


import diffrax as dfx

def x(t):  # control signal
    return t + 1

# exponential decay subject to affine control
def vector_field(t, y, args):
    return -y + x(t)

term = dfx.ODETerm(vector_field)
solver = dfx.Tsit5()
dfx.diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=1)


# def test_signal(t):
#     return t + 1

# def 