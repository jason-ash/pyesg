"""Create the pyesg logo using matplotlib"""
from matplotlib import pyplot as plt  # type: ignore # pylint: disable=import-error
from pyesg import GeometricBrownianMotion


model = GeometricBrownianMotion(mu=0.05, sigma=0.2)
lines = model.scenarios(
    100.0, dt=1 / 252, n_scenarios=10, n_steps=252, random_state=42
).T


fig, ax = plt.subplots(figsize=(5, 2))
fig.patch.set_facecolor("w")
plt.tight_layout(0)

ax.plot(lines)
ax.annotate(
    "pyesg", xy=(126, 160), ha="center", va="bottom", color="#0D47A1", fontsize=84
)

ax.set_xlim(0, 253)
ax.set_ylim(75, 165)
ax.set_yticklabels("")
ax.set_xticklabels("")
ax.tick_params(axis="both", colors="0.8")
ax.spines["bottom"].set_color("0.8")
ax.spines["left"].set_color("0.8")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.savefig("images/pyesg.png", bbox_inches="tight", dpi=300)
