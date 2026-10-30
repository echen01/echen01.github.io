"""Evaluate the unmodified upstream JAX step_frame and its log-area gradient.

Input: JSON cases on stdin. Output: JSON reference values on stdout.
Pass --upstream to an existing checkout at the pinned commit, or omit it to
download just the three required source files into a temporary directory.
Every source file is verified against its upstream Git blob hash before import.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import tempfile
import urllib.request

COMMIT = "b05116e06afbd7a569f3b9ac1d03a52c9d18efed"
BLOBS = {
    "__init__.py": "6a6527412913acb5cbdb4cb7c77e79647c2179bb",
    "tract.py": "c54b621a3b5c1ecc1c10aecb092e0cf860060fc6",
    "perlin.py": "8f78d8b500c00fd0d237e58d9fed0bb032af13e4",
}


def source_tree(root):
    for name, expected in BLOBS.items():
        path = root / "difftlm" / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/echen01/diffTLM/{COMMIT}/difftlm/{name}"
            with urllib.request.urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        data = path.read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest != expected:
            raise ValueError(f"{path} does not match upstream commit {COMMIT}")
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path)
    args = parser.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix="difftlm-jax-source-")
    root = source_tree(args.upstream or Path(temporary.name))
    sys.path.insert(0, str(root))

    import jax
    # The browser model uses f64. Configure before constructing any JAX arrays.
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")
    import jax.numpy as jnp
    import jaxlib
    import numpy as np
    import scipy
    from difftlm.tract import SimulatorOptions, TransmissionLineSimulator

    cases = json.load(sys.stdin)
    evaluators = {}
    results = []
    for case in cases:
        key = (case["sampleRate"], case["fftSize"])
        if key not in evaluators:
            sr, size = key
            simulator = TransmissionLineSimulator(SimulatorOptions(SR=sr, N=32, TF=1, fft_size=size))
            impulse = jnp.zeros(size, dtype=jnp.float64).at[0].set(1)
            noise = jnp.zeros((31, size), dtype=jnp.float64)

            # Default arguments capture this simulator/grid, including when the
            # next case constructs a different evaluator. No acoustic equations
            # are reimplemented here: this calls the hash-verified upstream code.
            def response(theta, length, sim=simulator, u=impulse, n=noise):
                return jnp.fft.rfft(sim.step_frame(u, jnp.exp(theta), n, length))

            def loss(theta, target_db, bins, length, centered=False, response_fn=response):
                h = response_fn(theta, length)[bins]
                target = target_db * jnp.log(10.0) / 20.0
                residual = jnp.log(jnp.abs(h)) - target
                if centered:
                    residual = residual - jnp.mean(residual)
                return jnp.mean(residual ** 2)

            evaluators[key] = (jax.jit(response), jax.jit(jax.value_and_grad(loss)),
                jax.jit(jax.value_and_grad(lambda theta, target, bins, length, loss_fn=loss: loss_fn(theta, target, bins, length, True))))

        response_fn, value_and_grad, centered_value_and_grad = evaluators[key]
        theta = jnp.log(jnp.asarray(case["areas"], dtype=jnp.float64))
        target_theta = jnp.log(jnp.asarray(case["targetAreas"], dtype=jnp.float64))
        bins = jnp.asarray(case["bins"], dtype=jnp.int32)
        length = jnp.asarray(case["length"], dtype=jnp.float64)
        if min(case["bins"]) <= 0 or max(case["bins"]) >= case["fftSize"] // 2:
            raise ValueError("Use interior bins: irfft projects DC and Nyquist onto the real axis.")
        h = np.asarray(response_fn(theta, length)[bins])
        target_h = response_fn(target_theta, length)[bins]
        target_db = 20.0 / jnp.log(10.0) * jnp.log(jnp.abs(target_h))
        value, gradient = value_and_grad(theta, target_db, bins, length)
        gradient = np.asarray(gradient)
        centered_value, centered_gradient = centered_value_and_grad(theta, target_db, bins, length)
        gain_db = jnp.mean(target_db - 20.0 / jnp.log(10.0) * jnp.log(jnp.abs(h)))
        assert h.dtype == np.complex128 and gradient.dtype == np.float64
        assert np.all(np.isfinite(h)) and np.all(np.isfinite(gradient)) and np.isfinite(float(value))
        results.append({
            "name": case["name"],
            "frequencies": [b * case["sampleRate"] / case["fftSize"] for b in case["bins"]],
            "real": h.real.tolist(), "imag": h.imag.tolist(),
            "targetDb": np.asarray(target_db).tolist(),
            "loss": float(value), "gradient": gradient.tolist(),
            "gainInvariantLoss": float(centered_value), "gainInvariantGradient": np.asarray(centered_gradient).tolist(),
            "gainDb": float(gain_db),
        })
        print(f"JAX: {case['name']} ({len(case['bins'])} bins)", file=sys.stderr, flush=True)

    json.dump({
        "upstreamCommit": COMMIT, "sourceGitBlobs": BLOBS,
        "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "backend": jax.default_backend(), "x64": jax.config.x64_enabled,
        "reference": "Unmodified TransmissionLineSimulator.step_frame; impulse input; zero noise; rfft; jax.value_and_grad with respect to log areas",
        "cases": results,
    }, sys.stdout, allow_nan=False)
    temporary.cleanup()


if __name__ == "__main__":
    main()
