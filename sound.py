import numpy as np
import librosa
import matplotlib.pyplot as plt

# =========================
# Plot spectrum
# =========================
def plot_spectrum(file_path, title="Spectrum"):
    y, sr = librosa.load(file_path, sr=None)

    # Windowing
    window = np.hanning(len(y))
    y = y * window

    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(Y), 1 / sr)

    mask = freqs > 0
    freqs = freqs[mask]
    magnitude = np.abs(Y[mask])

    # Normalise
    magnitude = magnitude / np.max(magnitude)

    plt.figure()
    plt.plot(freqs, magnitude)
    plt.xlim(0, 2000)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalised Magnitude")
    plt.show()


# =========================
# Plot waveform + sustain
# =========================
def plot_waveform(file_path, title="Waveform"):
    y, sr = librosa.load(file_path, sr=None)
    t = np.arange(len(y)) / sr

    energy = y**2
    energy = energy / np.max(energy)

    plt.figure()
    plt.plot(t, energy)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalised Energy")
    plt.show()


# =========================
# Extract timbre features
# =========================
def extract_timbre_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Windowing
    window = np.hanning(len(y))
    y = y * window

    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(Y), 1 / sr)

    mask = freqs > 0
    freqs = freqs[mask]
    magnitude = np.abs(Y[mask])

    peak_idx = np.argmax(magnitude)
    f0 = freqs[peak_idx]

    magnitude = magnitude / np.max(magnitude)

    A = np.sum(freqs * magnitude) / np.sum(magnitude)
    S = magnitude[peak_idx]

    harmonics = freqs / f0
    H = np.mean(np.abs(harmonics - np.round(harmonics)))

    M = np.mean(np.diff(magnitude))
    MA = np.mean(np.abs(freqs - np.mean(freqs))) / f0
    MC = np.mean(np.abs(magnitude - S))

    energy = y**2
    energy = energy / np.max(energy)

    indices = np.where(energy > 0.1)[0]
    sustain = (indices[-1] - indices[0]) / sr if len(indices) > 0 else 0

    feature_vector = np.array([f0, A, S, H, M, MA, MC, sustain])
    return feature_vector


# =========================
# Timbre distance
# =========================
def timbre_distance(f1, f2):
    return np.linalg.norm(f1 - f2)


# =========================
# Compare instruments
# =========================
def compare_instruments(files):
    features = {}

    for name, path in files.items():
        features[name] = extract_timbre_features(path)
        print(f"{name} features:\n{features[name]}\n")

    print("=== Timbre Distance ===")
    names = list(features.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = timbre_distance(features[names[i]], features[names[j]])
            print(f"{names[i]} vs {names[j]}: {d:.3f}")


files = {
    "3D_printed": "u1.wav",
    "Real_ukulele": "u2.wav"
}

compare_instruments(files)

plot_spectrum("u1.wav", "3D Printed Ukulele Spectrum")
plot_spectrum("u2.wav", "Real Ukulele Spectrum")
plot_waveform("u1.wav", "3D Printed Ukulele Sustain")
plot_waveform("u2.wav", "Real Ukulele Sustain")