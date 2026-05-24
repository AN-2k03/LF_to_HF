"""
NMR Spectrum Simulator — Quantum Mechanical 1H NMR
====================================================
  H = H_CS + H_J
  H_CS = sum_i (omega_i * Iz_i)          [chemical shift Hamiltonian]
  H_J  = sum_{i<j} 2*pi*J_ij * (Ix_i*Ix_j + Iy_i*Iy_j + Iz_i*Iz_j)  [J-coupling]

Frequencies of transitions are eigenvalue differences of H.
Intensities come from the off-diagonal elements of Ix+Iy (raising operator).

Spectral dimensions:
  SWD  : spectral width in the DIRECT dimension (Hz)  — the "observed" axis
  NPD  : number of time-domain points (FID samples) in the direct dimension
  dwell time  dt = 1/SWD   (Nyquist)
  frequency resolution  df = SWD/NPD
"""

import numpy as np
import copy
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#  SPIN-OPERATOR BUILDER
# ─────────────────────────────────────────────

def n_spin(n):
    """
    Build single-spin and product-operator matrices for an n-spin-1/2 system.

    For a single spin:
        sx, sy, sz  are the standard 2x2 Pauli/2 matrices.

    For n spins the Hilbert space has dimension 2^n.
    Each spin-k operator acts on spin k and is identity on all others:
        Iz_k = I ⊗ I ⊗ ... ⊗ sz ⊗ ... ⊗ I   (sz sits at position k)

    Returns
    -------
    x_list, y_list, z_list  — each a list of n matrices of shape (2^n, 2^n)
    """
    sx = 0.5 * np.array([[0, 1],  [1, 0]],    dtype=complex)   # Ix
    sy = 0.5 * np.array([[0,-1j], [1j, 0]],   dtype=complex)   # Iy
    sz = 0.5 * np.array([[1, 0],  [0, -1]],   dtype=complex)   # Iz
    E  = np.eye(2, dtype=complex)

    if n == 1:
        # trivial: just return the single-spin operators as 1-element lists
        return [sx], [sy], [sz]

    # ── bootstrap with n=2 ──────────────────────────────────────────────────
    E_list = [np.kron(E, E),  np.kron(E, E)]
    x_list = [np.kron(sx, E), np.kron(E, sx)]
    y_list = [np.kron(sy, E), np.kron(E, sy)]
    z_list = [np.kron(sz, E), np.kron(E, sz)]

    # ── extend iteratively for n > 2 ────────────────────────────────────────
    for i in range(3, n + 1):
        E_list2 = copy.copy(E_list)
        x_list2 = copy.copy(x_list)
        y_list2 = copy.copy(y_list)
        z_list2 = copy.copy(z_list)

        for j in range(i - 1):
            if j == 0:
                # expand the FIRST existing operator by tensoring E on the right
                E_list[0] = np.kron(E_list2[j], E)
                x_list[0] = np.kron(x_list2[j], E)
                y_list[0] = np.kron(y_list2[j], E)
                z_list[0] = np.kron(z_list2[j], E)

            # build the NEW spin-i operator: E ⊗ (old spin-j operator)
            Ek = np.kron(E, E_list2[j])
            xk = np.kron(E, x_list2[j])
            yk = np.kron(E, y_list2[j])
            zk = np.kron(E, z_list2[j])

            if len(E_list) > j + 1:
                E_list[j + 1] = Ek
                x_list[j + 1] = xk
                y_list[j + 1] = yk
                z_list[j + 1] = zk
            else:
                E_list.append(Ek)
                x_list.append(xk)
                y_list.append(yk)
                z_list.append(zk)

    return x_list, y_list, z_list


#hamiltonian builder

def build_hamiltonian(offsets_hz, couplings, x_list, y_list, z_list):
    """
    H = H_CS + H_J   (in angular-frequency units, rad/s)

    H_CS  =  sum_i  2π * offset_i * Iz_i
             (offset_i = chemical-shift frequency in Hz relative to carrier)

    H_J   =  sum_{(i,j)} 2π * J_ij * (Ix_i·Ix_j + Iy_i·Iy_j + Iz_i·Iz_j)
             The scalar product gives the isotropic J-coupling (weak or strong).

    Parameters
    ----------
    offsets_hz : list of float  — offset from carrier in Hz for each spin
    couplings  : list of (i, j, J_hz) tuples  — 0-based spin indices, J in Hz
    x/y/z_list : operator lists from n_spin()
    """
    dim = x_list[0].shape[0]
    H = np.zeros((dim, dim), dtype=complex)

    # Chemical shift
    for k, off in enumerate(offsets_hz):
        H += 2 * np.pi * off * z_list[k]

    # J-coupling  (full scalar coupling, not just Iz·Iz)
    for (i, j, J) in couplings:
        H += (2 * np.pi * J) * (
            x_list[i] @ x_list[j] +
            y_list[i] @ y_list[j] +
            z_list[i] @ z_list[j]
        )

    return H



# spectrum from eigenvalues

def compute_transitions(H, x_list, y_list):
    """
    Diagonalise H, then find allowed transitions.

    Allowed transition |m> → |n> :
        frequency  ω_mn = E_m - E_n   (rad/s)
        intensity  ∝ |<m| Ix+Iy |n>|²   (detection via Ix + iIy, the raising op)

    Returns list of (freq_hz, intensity) for all transitions with intensity > threshold.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)   # eigh: H is Hermitian

    # Total raising operator  I+ = Ix + i·Iy  (summed over all spins)
    dim = x_list[0].shape[0]
    I_raise = np.zeros((dim, dim), dtype=complex)
    for k in range(len(x_list)):
        I_raise += x_list[k] + 1j * y_list[k]   # Ix_k + i·Iy_k

    # Transform detection operator into eigenbasis
    I_raise_eig = eigenvectors.conj().T @ I_raise @ eigenvectors

    transitions = []
    n_states = len(eigenvalues)
    for m in range(n_states):
        for n in range(n_states):
            if m == n:
                continue
            intensity = abs(I_raise_eig[m, n]) ** 2
            if intensity > 1e-6:   # drop negligibly weak lines
                freq_hz = (eigenvalues[m] - eigenvalues[n]) / (2 * np.pi)
                transitions.append((freq_hz, intensity))

    return transitions


#  FID & FFT (direct dimension)

def simulate_fid(transitions, NPD, SWD, lw_hz=1.0):
    """
    Build a time-domain FID from the stick spectrum, then FFT.

    SWD  (spectral width, direct dimension, Hz)
      Sets the acquisition bandwidth.  dwell time dt = 1/SWD.
      Determines the highest frequency resolvable (Nyquist = SWD/2).

    NPD  (number of points, direct dimension)
      Length of the digitised FID.
      Frequency resolution after FFT:  df = SWD / NPD  (Hz/point).
      Longer NPD → finer frequency resolution in the spectrum.

    lw_hz : Lorentzian linewidth (Hz) — T2 relaxation broadening
            T2 = 1/(π·lw_hz)
    """
    dt = 1.0 / SWD                      # dwell time (s)
    t  = np.arange(NPD) * dt            # time axis

    fid = np.zeros(NPD, dtype=complex)
    T2  = 1.0 / (np.pi * lw_hz)

    for (freq_hz, intensity) in transitions:
        # Each line contributes a damped sinusoid to the FID
        fid += intensity * np.exp(1j * 2 * np.pi * freq_hz * t) * np.exp(-t / T2)

    # FFT → spectrum  (fftshift centres zero-frequency)
    spectrum = np.fft.fftshift(np.fft.fft(fid))
    freq_axis = np.fft.fftshift(np.fft.fftfreq(NPD, d=dt))   # Hz axis

    return freq_axis, np.real(spectrum)

#building the ppm axis 

def hz_to_ppm(freq_hz_axis, B0_MHz, carrier_ppm=4.7):
    """
    Convert a frequency axis (Hz, relative to carrier) to ppm.

    ppm = carrier_ppm + freq_Hz / B0_MHz

    The carrier is set to the water resonance at 4.7 ppm (standard for 1H).
    B0_MHz is the proton Larmor frequency (e.g. 400 for a 400 MHz magnet).
    """
    return carrier_ppm + freq_hz_axis / B0_MHz

def main():
    # ── Spin system ───────────────────────────────────────────────────────
    n_spins = int(input("Number of spins: "))
    shifts_ppm = []
    for k in range(n_spins):
        shifts_ppm.append(float(input(f"  Spin {k+1} chemical shift (ppm): ")))

    couplings = []
    has_coupling = input("J-couplings present? (Y/N): ").strip().upper()
    if has_coupling == "Y":
        coupled_pairs = []
        while True:
            entry = input("  Coupled pair (or 'done'): ").strip()
            if entry.lower() == "done":
                break
            a, b = [int(x.strip()) for x in entry.split(",")]
            coupled_pairs.append((a, b))
        for (a, b) in coupled_pairs:
            J = float(input(f"  J({a},{b}) in Hz: "))
            couplings.append((a-1, b-1, J))

    SWD_ppm = float(input("Spectral width (ppm) (e.g. 10): "))  # e.g. 10 ppm
    lw_hz = float(input("Linewidth (Hz): "))

    # accepting multiple B0 values 
    b0_input = input("Enter B0 values in MHz, comma-separated (e.g. 400,600,900): ")
    B0_list  = [float(x.strip()) for x in b0_input.split(",")]

    # compute time taken for the simulation
    import time
    start = time.time()
    
    NPD   = 1028*16
    carrier_ppm = 4.7  # water resonance as carrier frequency (ppm)
    #Plotting 

    fig1, ax1 = plt.subplots(figsize=(13, 5))
    fig2, ax2 = plt.subplots(figsize=(13, 5))
    colors = plt.cm.tab10.colors

    for idx, B0_MHz in enumerate(B0_list):
        SWD = SWD_ppm * B0_MHz
        offsets_hz = [(p - carrier_ppm) * B0_MHz for p in shifts_ppm]

        x_list, y_list, z_list = n_spin(n_spins)
        H = build_hamiltonian(offsets_hz, couplings, x_list, y_list, z_list)
        transitions = compute_transitions(H, x_list, y_list)
        freq_axis_hz, spectrum = simulate_fid(transitions, NPD, SWD, lw_hz)
        ppm_axis = hz_to_ppm(freq_axis_hz, B0_MHz, carrier_ppm)

        spectrum_norm = spectrum / np.trapezoid(np.abs(spectrum))

        color = colors[idx % len(colors)]
        label = f"{B0_MHz:.0f} MHz"

        # top plot: ppm axis
        #ax1.plot(ppm_axis, spectrum_norm + idx * -0.009, color=color, linewidth=0.9, label=label)
        ax1.plot(ppm_axis, spectrum_norm, color=color, linewidth=0.9, label=label)

        # bottom plot: Hz axis (offset from carrier)
        ax2.plot(freq_axis_hz, spectrum_norm + idx * -0.009, color=color, linewidth=0.9, label=label)

    # top plot formatting
    ax1.set_xlabel("Chemical Shift (ppm)")
    ax1.set_ylabel("Intensity (normalised)")
    ax1.set_title("ppm axis") #peak positions invariant to B0
    ax1.invert_xaxis()
    half_ppm = SWD_ppm / 2
    ax1.set_xlim([carrier_ppm + half_ppm, carrier_ppm - half_ppm])
    ax1.set_xticks(np.arange(round(carrier_ppm - half_ppm), round(carrier_ppm + half_ppm) + 1, 1))
    ax1.legend(title="B0")
    ax1.set_yticks([])

    # bottom plot formatting
    ax2.set_xlabel("Frequency (Hz from carrier)")
    ax2.set_ylabel("Intensity (normalised)")
    ax2.set_title("Hz axis") # peaks shift with B0, but J-coupling spacing stays fixed
    ax2.invert_xaxis()
    ax2.set_xlim([SWD/2, -SWD/2])
    ax2.legend(title="B0")
    ax2.set_yticks([])

    fig1.tight_layout()
    fig2.tight_layout()
    end = time.time()
    print(f"\nTotal simulation time: {end - start:.2f} seconds")
    plt.show()

if __name__ == "__main__":
    main()

