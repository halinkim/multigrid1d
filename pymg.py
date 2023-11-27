class Multigrid:
    def __init__(self, depth, iter, x0, x1, u0, u1, s=None):
        self.depth = depth
        self.N = 2 ** depth
        self.iter = iter
        self.x0 = -1
        self.x1 = 1

        self.num_cells = [(2 << _) for _ in range(depth)]
        self.dx = [(x1 - x0) / (2 << _) for _ in range(depth)]
        self.x = [[x0 + i * self.dx[_] for i in range((2 << _) + 1)] for _ in range(depth)]
        self.u = [[0] * ((2 << _) + 1) for _ in range(depth)]
        self.s = [[0] * ((2 << _) + 1) for _ in range(depth)]
        self.res = [[0] * ((2 << _) + 1) for _ in range(depth)]
        self.u[-1][0] = u0
        self.u[-1][-1] = u1
        if s:
            for i in range(1, self.N):
                self.s[-1][i] = s[i]

    def relax(self, depth):
        for _ in range(self.iter):
            self.relax_rb(depth)

    def relax_gs(self, depth):
        for i in range(1, self.num_cells[depth]):
            self.u[depth][i] = (self.u[depth][i + 1] + self.u[depth][i - 1] - self.s[depth][i] * self.dx[depth] *
                                self.dx[depth]) / 2

    def relax_rb(self, depth):
        for i in range(1, self.num_cells[depth], 2):
            self.u[depth][i] = (self.u[depth][i + 1] + self.u[depth][i - 1] - self.s[depth][i] * self.dx[depth] *
                                self.dx[depth]) / 2
        for i in range(2, self.num_cells[depth], 2):
            self.u[depth][i] = (self.u[depth][i + 1] + self.u[depth][i - 1] - self.s[depth][i] * self.dx[depth] *
                                self.dx[depth]) / 2

    def prolongation(self, depth):
        for i in range(self.num_cells[depth]):
            if i & 1:
                self.u[depth][i] += (self.u[depth - 1][i >> 1] + self.u[depth - 1][(i >> 1) + 1]) / 2
            else:
                self.u[depth][i] += self.u[depth - 1][i >> 1]

    def restriction(self, depth):
        self.s[depth][0] = self.res[depth + 1][0]
        self.s[depth][-1] = self.res[depth + 1][-1]
        for i in range(1, self.num_cells[depth]):
            self.s[depth][i] = (self.res[depth + 1][2 * i - 1] + 2 * self.res[depth + 1][2 * i] + self.res[depth + 1][
                2 * i + 1]) / 4
        for i in range(self.num_cells[depth]):
            self.u[depth][i] = 0

    def residual(self, depth):
        for i in range(1, self.num_cells[depth]):
            self.res[depth][i] = self.s[depth][i] - (
                        self.u[depth][i + 1] + self.u[depth][i - 1] - 2 * self.u[depth][i]) / self.dx[depth] / self.dx[
                                     depth]

    def norm_residual(self):
        cnt = 0
        for i in range(self.N):
            cnt += self.res[self.depth - 1][i] ** 2
        cnt /= self.N
        return cnt ** .5 * self.dx[-1] * self.dx[-1]

    def multigrid(self, lvs):
        if lvs == 0:
            self.relax(lvs)
            return
        self.relax(lvs)
        self.residual(lvs)
        self.restriction(lvs - 1)
        self.multigrid(lvs - 1)
        self.prolongation(lvs)

        self.relax(lvs)

    def solve(self):
        for _ in range(10):
            self.multigrid(self.depth - 1)
            x = self.norm_residual()
            if x < 1E-15:
                break
            # print(x)

        return self.u[-1][:]

def norm2res(seq):
    cnt = 0
    for i in seq:
        cnt += i * i
    cnt /= len(seq)
    return cnt ** .5

if __name__ == '__main__':
    import math

    def src_func(x):
        return - math.pi**2 * math.sin(math.pi * x)
    x0 = 0
    x1 = 1
    u0 = 1
    u1 = 0
    level = 19
    N = 1 << level
    dx = (x1 - x0) / N
    x = [x0 + i * dx for i in range(N + 1)]
    source = [src_func(i) for i in x]
    mg = Multigrid(level, 1, x0, x1, u0, u1, source)
    mg.solve()