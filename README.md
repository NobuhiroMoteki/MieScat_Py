# MieScat_Py

## 📌 Description
Python codes for calculating scattering and absorption of plane-electromegnetic waves by a isotropic homogeneous sphere, aka the Lorenz-Mie theory.

### List of codes
- `miescat.py` computes the optical efficiency factors (Qsca, Qext, Qabs, and Qback), scattering amplitudes (S11 and S22), and optical cross sections per unit mass (MSC, MEC, and MAC).
- `mie_complex_amplitudes.py` computes the scattering amplitudes observable by the Complex Amplitude Sensing (CAS-v1 and CAS-v2) 
- `call_mie_scattering.ipynb` demonstrates how to use these Mie scattering codes.

---

## 🚀 Installation

The author developed and tested current version (v0.1.1) using Python 3.12.8 in Windows 11 machines.

#### 1. Clone the repository
```sh
git clone https://github.com/NobuhiroMoteki/MieScat_Py.git
cd MieScat_Py
```

#### 2. Install dependencies
```sh
pip install -r requirements.txt
```
---

## 🔧 Usage

Please see the comments in each codes.

---

## 📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 📖 References
- Mie theory and its computational methods
    - Bohren, C. F., & Huffman, D. R. (2008). Absorption and scattering of light by small particles. John Wiley & Sons.
    - Mishchenko, M. I., Travis, L. D., & Lacis, A. A. (2002). Scattering, absorption, and emission of light by small particles. Cambridge university press.
  
- Complex Amplitude Sensing (CAS)
    - Moteki, N. (2021). Measuring the complex forward-scattering amplitude of single particles by self-reference interferometry: CAS-v1 protocol. Optics Express, 29(13), 20688-20714.
    - Moteki, N., & Adachi, K. (2024). Measuring the polarized complex forward-scattering amplitudes of single particles in unbounded fluid flow: CAS-v2 protocol. Optics Express, 32(21), 36500-36522.



## 📢 Author
Name: Nobuhiro Moteki
GitHub: @NobuhiroMoteki
Email: nobuhiro.moteki@gmail.com


