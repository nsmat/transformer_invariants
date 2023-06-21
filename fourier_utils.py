import numpy as np

class Complex:

    def __init__(self, r, theta):
        assert r >= 0
        self.r = r
        self.theta = theta % (np.pi*2)

    def __repr__(self):
        return f"{self.r}*exp[i {self.theta}]"

    @property
    def euc_form(self):
        # Python natively uses Euclidean form
        return self.r*np.exp(1j*self.theta)

    @property
    def real(self):
        return self.r * np.cos(self.theta)

    @property
    def im(self):
        return self.r * np.sin(self.theta)

    @classmethod
    def from_euc(cls, z):
        r = abs(z)
        theta = np.arctan2(z.imag, z.real)

        return cls(r, theta)


def fourier_transform_point_function(z: Complex, precision):
    """Analytically obtain the fourier transform of a dirac function on S1 """
    indexes = np.arange(-precision, precision + 1)
    c = [
        z.r*np.exp(-1j*k*z.theta) for k in indexes # cast to euc form so python can do operations
    ]
    scaler = 1/(2*np.pi)
    c = np.array(c) * scaler
    f_hat = np.column_stack((indexes, c))
    return f_hat

def inverse_transform(f_hat):
    k = f_hat[:, 0]
    c = f_hat[:, 1]

    scaler = 1/np.pi
    components = lambda x: c*np.exp(-k*x*1j)
    f = lambda x: scaler*components(x).sum()
    f = np.vectorize(f)
    return f