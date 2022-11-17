import diffrax as dfx


class SimManager():
    """ Handle the running of a single simulation. Allows combination of 
    different functions """

    def __init__(self, solver, saveat) -> None:
        self.solver = dfx.Tsit5()
        self.saveat = dfx.SaveAt(t0=True, t1=True, steps=True)
        self.max_steps = 16**4

    def run(self, term, t0, t1, dt0, y0):

        return dfx.diffeqsolve(term, self.solver, t0=t0, t1=t1, dt0=dt0,
                             y0=y0,
                             saveat=self.saveat, max_steps=self.max_steps)
