# willychan.github.io

Super simple personal GitHub Pages site.

- **Home** — `index.html`
- **Projects** — Add a `.md` file in `_projects/` for each project. Front matter: `title`, `image` (URL), `description`, optional `link` (URL), optional `recognition` (list of bullets). They appear as boxes on the Projects page.
- **Blog** — Add new posts as `.md` files in `_posts/` with the filename format: `YYYY-MM-DD-title.md` (each file needs `layout: post` and `title: Your title` at the top)

Push to the `main` branch and GitHub Pages will build it (Jekyll runs automatically).

### Preview locally (without pushing)

**Option A — Docker (works with any Ruby):**
```bash
docker run --rm -v "$(pwd)":/site -p 4000:4000 jekyll/jekyll jekyll serve --force_polling
```
Then open **http://localhost:4000**

**Option B — Ruby 3+ and Bundler:**
```bash
bundle install && bundle exec jekyll serve
```
Then open **http://localhost:4000**
