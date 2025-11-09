import matplotlib.pyplot as plt
import pandas as pd

times = [30, 80, 300]

fig, axs = plt.subplots(3, 2, figsize=(12, 9))
for i, t_ms in enumerate(times):
    fname = f"t_{t_ms}.csv"
    data = pd.read_csv(fname)

    
    axs[i, 0].plot(data["x"], data["numerical"], label="Numerical", color="blue")
    axs[i, 0].plot(data["x"], data["analytical"], "--", label="Analytical", color="orange")
    axs[i, 0].set_title(f"t = {t_ms/1000:.2f}")
    axs[i, 0].set_ylabel("u(x,t)")
    axs[i, 0].grid(True)
    if i == 0:
        axs[i, 0].legend()


    axs[i, 1].plot(data["x"], data["abs_error"], color="red")
    axs[i, 1].set_title(f"Error at t = {t_ms/1000:.2f}")
    axs[i, 1].grid(True)

axs[-1, 0].set_xlabel("x")
axs[-1, 1].set_xlabel("x")

plt.tight_layout()
plt.show()
