"""Independent NumPy reference for the current JAX stationary tract equations.

Uses full 2x2 complex matrix products, rather than the Rust first-column update.
Run this when changing physics; test.mjs compares the WebAssembly result.
"""
import json
import numpy as np


def response(areas, length, frequencies):
    rho, c, mu = 1.14e-3, 3.43e4, 1.86e-4
    areas = np.asarray(areas, dtype=np.float64)
    n = len(areas)
    lr = 8 * rho / (3 * np.pi * np.sqrt(np.pi * areas[-1]))
    rr = 128 * rho * c / (9 * np.pi**2 * areas[-1])
    a = areas[:-1]
    li = length / n
    sa = np.sqrt(a / np.pi) * 2 * np.pi * li
    inductance = rho * li / (2 * a)
    resistance = sa / (2 * a**2) * np.sqrt(rho * mu * 0.5)
    capacitance = a * li / (rho * c**2)
    lw, rw, cw = 2.1 / sa, 800 / sa, sa / 84500
    omega = 2 * np.pi * np.maximum(frequencies, 1e-8)
    iw = 1j * omega[:, None]
    za = resistance * np.sqrt(omega[:, None]) + inductance * iw
    zb1 = rw + lw * iw + 1 / (iw * cw)
    zb2 = 1 / (iw * capacitance)
    zb = zb1 * zb2 / (zb1 + zb2)
    zr = rr * (1j * omega * lr) / (rr + 1j * omega * lr)
    matrices = np.stack([1 + za / zb, -2 * za - za**2 / zb, -1 / zb, 1 + za / zb], axis=-1).reshape(len(omega), -1, 2, 2)
    product = matrices[:, 0]
    for i in range(1, n - 1):
        product = matrices[:, i] @ product
    h = zr / (product[:, 0, 0] - product[:, 1, 0] * zr)
    return {"real": h.real.tolist(), "imag": h.imag.tolist()}


frequencies = np.array([0, 80, 123, 500, 1017, 2331, 4000, 5000, 7999, 16000], dtype=float)
cases = []
for area, length in [(np.ones(32) * 0.12, 12), (np.ones(32) * 3, 17.5), (np.ones(32) * 16, 22), (np.exp(np.linspace(-1.5, 2, 32)), 17.5), (2 + np.sin(np.arange(32) * 0.38), 20)]:
    cases.append({"areas": area.tolist(), "length": length, "frequencies": frequencies.tolist(), **response(area, length, frequencies)})
print(json.dumps(cases))
