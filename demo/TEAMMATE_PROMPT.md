You are helping me publish my team's website today; the deadline is 20:00 (Kazakhstan time, UTC+5). Work step by step,
tell me exactly what to click when something must be done in a web dashboard, and never ask me for passwords,
API tokens or 2FA codes — I sign in myself. Do not change anything that is not listed below.

## What exists already
- **Site files:** `qyraneye_site.zip` (I will give you its path). Unzipped it is a folder `site/` with `index.html`
  and `examples/` (3 mp4 videos + 3 jpg posters, ~28 MB total). It is a static page: no build step, no framework.
- **Processing backend (already running, do not touch):** https://zhangir954--qyraneye-ui.us-east.modal.run —
  a Gradio app on Modal. `index.html` embeds it in an `<iframe>` and links to it. It may take ~30 s to wake up.
- **Cloudflare account** (I am signed in in my browser): Workers & Pages has a Pages project **`conrad-qyran`**
  with no deployments yet. There is another project `cis-hub-qyran` — **do not touch it.**
- **Domain:** `qyraneye.space`, registered at **Namecheap**, already added to Cloudflare as a zone but
  **not active yet** (Cloudflare is waiting for the nameserver change). Cloudflare assigned these nameservers:
  `adel.ns.cloudflare.com` and `vern.ns.cloudflare.com`.

## Goal
1. The site is live on the `conrad-qyran` Pages project (address like `https://conrad-qyran.pages.dev`).
2. `https://qyraneye.space` (and `https://www.qyraneye.space`) show the same site over HTTPS.

## Steps
1. **Check the files.** Unzip, confirm `site/index.html` and the 6 files in `site/examples/` exist and the
   videos play locally. Do not edit the page unless something is broken; if you must, tell me what and why first.
2. **Deploy to Pages** (pick one):
   - Dashboard (simplest): Workers & Pages → `conrad-qyran` → *Upload assets* → drag the **`site` folder**
     (the folder itself, so `index.html` is at the root) → *Deploy site*.
   - Or CLI, if Node is installed: `npx wrangler login` (I approve in the browser), then
     `npx wrangler pages deploy site --project-name conrad-qyran --branch main`.
3. **Verify the pages.dev address:** the page loads; the 3 example videos play; the "Try it" section shows the
   QyranEye app inside the frame (wait up to 30 s); on a phone the layout fits the screen.
4. **Switch the nameservers at Namecheap** (I do the clicks, you guide me): Domain List → `qyraneye.space` →
   *Manage* → *Nameservers* → *Custom DNS* → `adel.ns.cloudflare.com`, `vern.ns.cloudflare.com` → save (green tick).
   If DNSSEC is ON at Namecheap, tell me to turn it off first (it breaks the switch).
   In Cloudflare, on the `qyraneye.space` overview, I can press *Check nameservers now*.
5. **Wait until the zone is Active** (usually 15 min – a few hours). Meanwhile the pages.dev link is the one to send.
6. **Attach the domain:** Pages → `conrad-qyran` → *Custom domains* → *Set up a custom domain* → `qyraneye.space`
   → let Cloudflare create the DNS record. Repeat for `www.qyraneye.space`. Wait until both show *Active* and SSL is
   issued.
7. **Final check from a phone on mobile data (not our Wi-Fi):** `https://qyraneye.space` loads, videos play, the
   app in "Try it" loads. Optionally upload a short (≤ 30 s) video there and confirm a result appears in 2–5 min.

## Rules
- Do not delete or change other projects, DNS zones, or the Modal app; do not create paid plans or add a card.
- If something fails, show me the exact error message and stop before trying risky changes.
- When done, give me: the pages.dev link, whether qyraneye.space is active yet, and anything left to do.
