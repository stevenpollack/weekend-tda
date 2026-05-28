# Publishing `pilot.ipynb` via GitHub Pages

Goal: render the notebook (code + outputs) as a static HTML page and serve it from GitHub Pages so anyone with the link can read the results without cloning the repo.

**Selected approach:** nbconvert + GitHub Actions (auto-deploy on every push to main).

---

## Steps

1. **Add `nbconvert` to dev deps:**
   ```bash
   uv add --group dev nbconvert
   ```

2. **Create `.github/workflows/pages.yml`:**
   ```yaml
   name: Publish notebook to GitHub Pages

   on:
     push:
       branches: [main]

   permissions:
     contents: read
     pages: write
     id-token: write

   concurrency:
     group: pages
     cancel-in-progress: true

   jobs:
     build:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4

         - name: Install uv
           uses: astral-sh/setup-uv@v4

         - name: Install deps
           run: uv sync --group dev

         - name: Convert notebook to HTML
           run: uv run jupyter nbconvert --to html pilot.ipynb --output _site/index.html

         - name: Upload Pages artifact
           uses: actions/upload-pages-artifact@v3
           with:
             path: _site

     deploy:
       needs: build
       runs-on: ubuntu-latest
       environment:
         name: github-pages
         url: ${{ steps.deployment.outputs.page_url }}
       steps:
         - name: Deploy to GitHub Pages
           id: deployment
           uses: actions/deploy-pages@v4
   ```

3. **Enable GitHub Pages in repo settings:**
   - Settings → Pages → Source: "GitHub Actions"

4. **Push to main.** The workflow converts and deploys automatically.

## Notes

- The notebook must be pre-executed (outputs saved in `.ipynb`) — the workflow does *not* re-run cells (that would require GPU, CIFAR download, etc.).
- Code cells are shown in the published HTML (no `--no-input` flag).
- Published URL will be `https://<username>.github.io/weekend-tda/`.
