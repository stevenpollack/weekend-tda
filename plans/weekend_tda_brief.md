# Weekend research brief: detecting coverage holes in encoder embeddings via persistent homology

**Junior researcher:** Steven
**PI:** Claude
**Time budget:** Saturday + Sunday, ~14 working hours total. Hard stop Sunday 6pm.

---

## 0. Read this first

This is **not** a research project. It is a calibration exercise. The goal is for you to:

1. Get hands-on with TDA tooling on real ML embeddings
2. Build intuition for what topological features look like in high dimensions
3. Discover the practical obstacles firsthand
4. Generate a calibrated opinion on whether the broader idea is worth pursuing further

You will likely produce ambiguous results. That is the expected outcome. **Do not oversell anything to yourself.** A clean negative ("TDA can't naively detect deliberately-induced coverage gaps in CLIP embeddings") is more valuable than a noisy positive.

---

## 1. Research question

**Original idea:** Empty regions in a model's latent space might encode meaningful absences from training data (intentional or otherwise).

**Sharpened to weekend-feasible version:**
> If we artificially remove one semantic class from a dataset before embedding it, can persistent homology detect a meaningful topological signature of that absence?

This is a *necessary* condition for the broader vision. If we can't detect deliberately-induced holes when we know they exist, we certainly can't detect natural ones.

---

## 2. The four phases

| Phase | When | Duration | Goal |
|-------|------|----------|------|
| 1. Setup + synthetic sanity check | Sat AM | 3h | Verify pipeline works on toy data |
| 2. Real embeddings baseline | Sat PM | 4h | CLIP + CIFAR-100, get embeddings, visualize |
| 3. Ablation experiment | Sun AM | 3h | Remove a class, compare persistence diagrams |
| 4. Writeup | Sun PM | 2h | One-page markdown summary, push to GitHub |

Buffer: 2h. You will need it.

---

## 3. Setup (Friday evening — 30 min, NOT counted in weekend budget)

Do this Friday night while your toddler is asleep. If it doesn't work Friday, you'll burn Saturday morning on dependencies instead of research.

```bash
mkdir tda-pilot && cd tda-pilot
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision open_clip_torch giotto-tda umap-learn matplotlib jupyter scikit-learn
```

Sanity check:
```python
import gtda, torch, open_clip, umap
print("ok")
```

If any of those fail, fix it Friday. **Do not start Phase 1 on Saturday with broken dependencies.**

Open a Jupyter notebook called `pilot.ipynb`. Initialize a git repo. First commit.

---

## 4. Phase 1: Synthetic sanity check (Saturday AM, 3h)

**Do not skip this.** If TDA can't detect holes you *know* are there in toy data, it certainly won't detect them in real data. This is also your chance to build intuition for persistence diagrams before you have any confounds.

### 4.1 Test case A: 2D circle with arc removed

```python
import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram

# Full circle
n = 200
theta = np.linspace(0, 2*np.pi, n, endpoint=False)
full = np.stack([np.cos(theta), np.sin(theta)], axis=1)

# Circle with 60-degree arc removed
mask = ~((theta > np.pi/2) & (theta < np.pi/2 + np.pi/3))
broken = full[mask]

vr = VietorisRipsPersistence(homology_dimensions=[0, 1])
diag_full = vr.fit_transform(full[None, ...])
diag_broken = vr.fit_transform(broken[None, ...])

plot_diagram(diag_full[0]).show()
plot_diagram(diag_broken[0]).show()
```

**What to look for:** Both should show one persistent H_1 feature (the loop). The broken one's H_1 feature should *die earlier* than the full one because the loop is "weaker" — there's a gap.

**Calibration question:** can you eyeball the persistence diagram and tell which one has the gap? If no, you don't yet have intuition. Spend more time here.

### 4.2 Test case B: Two clusters → one cluster

```python
np.random.seed(42)
cluster_a = np.random.randn(100, 10) + np.array([5]*10)
cluster_b = np.random.randn(100, 10) - np.array([5]*10)
both = np.vstack([cluster_a, cluster_b])
only_a = cluster_a

diag_both = vr.fit_transform(both[None, ...])
diag_a = vr.fit_transform(only_a[None, ...])
```

**What to look for:** `diag_both` should show 2 persistent H_0 features (two components). `diag_a` should show 1.

This is the easy case. If this doesn't work, your pipeline is broken.

### 4.3 Test case C: High-dim sphere with chunk removed

```python
# Sample uniformly on the 9-sphere in R^10
n = 500
pts = np.random.randn(n, 10)
pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

# Remove one "hemisphere" (e.g., points where first coordinate > 0.5)
mask = pts[:, 0] < 0.5
hemisphere_removed = pts[mask]

# Note: H_1 of S^9 is trivial — the meaningful feature would be H_9, but
# computing high-dim homology is expensive. We're really testing whether
# *anything* changes detectably.

vr_hd = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
diag_full_s = vr_hd.fit_transform(pts[None, ...])
diag_cut_s = vr_hd.fit_transform(hemisphere_removed[None, ...])
```

**What to look for:** This will probably be ambiguous. That's the *real* lesson — high-dim TDA is finicky. Note what you see.

### 4.4 Decision point

End of Phase 1 (lunch Saturday). Ask yourself:
- Did I understand the persistence diagrams?
- Did the toy ablation in 4.2 produce a clean expected result?

If **no** to both: stop. Spend the rest of the weekend reading the giotto-tda tutorials. Ship a writeup saying "did not get to real experiment, pipeline needed more learning."

If **yes**: proceed.

---

## 5. Phase 2: Real embeddings baseline (Saturday PM, 4h)

### 5.1 Load CLIP and embed CIFAR-100

```python
import open_clip
import torch
import torchvision
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device).eval()

dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=preprocess
)
loader = DataLoader(dataset, batch_size=128, shuffle=False)

all_emb, all_lab = [], []
with torch.no_grad():
    for imgs, labels in loader:
        emb = model.encode_image(imgs.to(device))
        emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize
        all_emb.append(emb.cpu().numpy())
        all_lab.append(labels.numpy())

embeddings = np.concatenate(all_emb)  # (10000, 512)
labels = np.concatenate(all_lab)
np.save('embeddings.npy', embeddings)
np.save('labels.npy', labels)
```

Save these — you don't want to recompute.

### 5.2 Visualize with UMAP

```python
import umap
reducer = umap.UMAP(n_components=2, random_state=42)
emb_2d = reducer.fit_transform(embeddings)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab20', s=2, alpha=0.5)
plt.title('CLIP embeddings of CIFAR-100 test set (UMAP)')
plt.savefig('umap_full.png', dpi=120)
```

You should see semantically meaningful clusters. This is your reference image.

### 5.3 Subsample for TDA

Full 10k × 512 is way too much for Vietoris-Rips. Subsample stratified:

```python
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, train_size=2000, random_state=42)
idx, _ = next(sss.split(embeddings, labels))
sub_emb = embeddings[idx]
sub_lab = labels[idx]
```

### 5.4 Decide on PCA preprocessing

In 512 dimensions, pairwise distances are dominated by noise (curse of dimensionality). You have two options:

- **A:** Run TDA on raw 512-dim embeddings
- **B:** PCA down to ~50 dims first

Run **both**. They will give different results. The disagreement is itself informative.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
sub_emb_pca = pca.fit_transform(sub_emb)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
```

### 5.5 Compute baseline persistence diagram

```python
vr_real = VietorisRipsPersistence(homology_dimensions=[0, 1], n_jobs=-1)
diag_baseline_512 = vr_real.fit_transform(sub_emb[None, ...])
diag_baseline_pca = vr_real.fit_transform(sub_emb_pca[None, ...])
plot_diagram(diag_baseline_512[0]).show()
plot_diagram(diag_baseline_pca[0]).show()
```

Expected runtime: 1–10 minutes each.

**If this takes more than 30 minutes, your point count is too high.** Drop to 1000 points. Don't burn your evening waiting for a single computation.

### 5.6 End of Saturday checkpoint

Commit to git. Write 3 sentences in your notebook describing what you see in the baseline diagrams. Take a break.

---

## 6. Phase 3: The ablation (Sunday AM, 3h)

### 6.1 Pick a class to remove

Use the CIFAR-100 class list. Pick something semantically distinctive — easier to reason about. Suggestions: `tank` (class 85), `tulip` (class 92), `tiger` (class 88). I'll arbitrarily say **tank** for concreteness.

```python
held_out = 85
mask = sub_lab != held_out
sub_emb_minus = sub_emb[mask]
sub_emb_minus_pca = sub_emb_pca[mask]
```

You've removed ~20 points (assuming roughly balanced stratified sample of 2000 across 100 classes). That's a 1% perturbation. Don't expect dramatic effects.

### 6.2 Compute ablated persistence diagrams

```python
diag_minus_512 = vr_real.fit_transform(sub_emb_minus[None, ...])
diag_minus_pca = vr_real.fit_transform(sub_emb_minus_pca[None, ...])
```

### 6.3 Compare

Side-by-side persistence diagrams: baseline vs. ablated, both for 512-d and PCA versions.

Also compute the **bottleneck distance** between baseline and ablated diagrams (giotto-tda has this) — this is the natural metric for comparing persistence diagrams.

```python
from gtda.diagrams import PairwiseDistance
pd = PairwiseDistance(metric='bottleneck')
# stack baseline and ablated diagrams to compare
```

### 6.4 Localize the "hole"

If you see a new persistent H_1 feature, can you find the points whose 2D UMAP projection sits at its center?

```python
# Where would the held-out tank class have been in UMAP space?
held_out_indices_in_full = np.where(labels == held_out)[0]
held_out_2d = emb_2d[held_out_indices_in_full]
# overlay these on the UMAP plot of the ablated set
```

This is the qualitative check: does the topological hole's location correspond to where the removed class lived in embedding space?

### 6.5 Realistic expected outcomes

Order of likelihood:
1. **(70%)** The two diagrams look nearly identical. Bottleneck distance is small. The 1% perturbation isn't detectable. **Lesson:** ablating one out of 100 classes from a 2000-point subsample probably doesn't produce a detectable topological signal at this scale.
2. **(20%)** There are differences, but they're hard to interpret and you can't cleanly attribute them to the removed class. **Lesson:** TDA is sensitive to many things; isolating one cause is hard.
3. **(10%)** Clean signal. Be skeptical — check your code for bugs, randomization, leakage.

If you get outcome 1: try a more aggressive ablation. Remove an entire **superclass** (CIFAR-100 has 20 superclasses of 5 classes each). That's 5% of the data and a semantically coherent chunk.

---

## 7. Phase 4: Writeup (Sunday PM, 2h)

Create `RESULTS.md` in the repo. **One page maximum.** Sections:

1. **What I ran** (1 short paragraph): dataset, model, ablation, tools.
2. **What I found** (1–2 paragraphs + 2–3 figures): baseline diagram, ablated diagram, UMAP showing where the held-out class lived.
3. **What didn't work / surprised me** (1 paragraph): be honest. Curse of dimensionality? Subsampling artifacts? Tooling friction?
4. **Should this be pursued further?** (1 paragraph): your calibrated take. Yes/no/conditional, with one reason.

Push to GitHub (public repo is fine; this is exploratory). Tell me the link.

---

## 8. Hard stop conditions

End the weekend at the **first** of:

- Sunday 6pm
- Phase 4 complete
- You've spent 3+ hours on a single technical issue (dependency hell, slow computation, mysterious bug)

If you hit the time limit before Phase 3 is done, **ship what you have**. Partial results are still results. A writeup saying "I got through Phase 2 and learned X, Y, Z about the tooling" is fine.

---

## 9. What NOT to do

- Do not try multiple encoders. One model. CLIP ViT-B/32.
- Do not try multiple datasets. CIFAR-100 only.
- Do not try multiple TDA libraries. giotto-tda only.
- Do not read papers during the weekend. Save them for after.
- Do not generalize from the result. One experiment, one model, one dataset is one data point.

---

## 10. Post-weekend reading (only after Phase 4)

If you're still interested after this exercise, in order:

- giotto-tda official tutorials (you'll have intuition for them now)
- Otter et al., "A roadmap for the computation of persistent homology" (2017) — the practitioner's guide
- Naitzat et al., "Topology of deep neural networks" (2020) — closest published work to your direction
- Carlsson, "Topology and data" (2009) — foundational, more mathematical

---

## 11. Final PI note

The most valuable output of this weekend is not the experiment. It is **your updated prior on whether this research direction is worth more of your finite time.** Update honestly. If the answer is "no," that's a successful weekend — you avoided spending months on it.

Good luck. Ship something by Sunday 6pm.
