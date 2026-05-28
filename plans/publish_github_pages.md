# Publishing `pilot.ipynb` via GitHub Pages

Goal: render the notebook (code + outputs) as a styled dark-mode HTML page via Quarto and deploy it to GitHub Pages on every push to main.

---

## How it works

1. `_quarto.yml` configures Quarto to render as a website with the `darkly` theme, outputting to `_site/`.
2. `.github/workflows/pages.yml` installs Quarto, runs `quarto render pilot.ipynb`, and deploys `_site/` via GitHub Pages.
3. The notebook must be pre-executed (outputs saved in `.ipynb`) — the workflow does not re-run cells.

## Local preview

```bash
quarto render pilot.ipynb
# open _site/pilot.html in browser
```

Or use `quarto preview pilot.ipynb` for live-reload during editing.

## Deployment

- Push to `main` triggers the workflow automatically.
- One-time setup: enable Pages in repo settings → Source: "GitHub Actions".
- Published URL: `https://<username>.github.io/weekend-tda/`

## Theme

Using Bootswatch `darkly`. Change in `_quarto.yml` under `format.html.theme`. Other dark options: `slate`, `cyborg`, `superhero`, `vapor`.
