"""
Creates the instaloader session file that ig_get_video.py / ig_scrape_post.py
/ ig_scrape_comment.py load via IG_SESSION_FILE, from an Instagram login
cookie exported out of a browser.

Actual workflow used for this project (Microsoft Edge; any browser with a
cookie-export extension works the same way):
1. Log into Instagram normally in Edge with the account you intend to
   scrape with.
2. Use a cookie-export extension (e.g. "Cookie-Editor") to export that
   session's cookies for instagram.com as JSON, overwriting
   edge_cookie.json (or wherever COOKIE_JSON_PATH points) with the fresh
   export -- a list of objects each with at least "name" and "value" keys.
3. Run this script (`python import_browser_session.py`) to turn that
   cookie export into the instaloader session file the other three
   scripts read via IG_SESSION_FILE.

Instagram sessions expire and get invalidated (e.g. after a period of
inactivity, or if the account is flagged); when ig_get_video.py /
ig_scrape_post.py / ig_scrape_comment.py start failing to log in, repeat
steps 1-3 to refresh the cookie export and regenerate the session file.

NEVER commit that cookie JSON file (or the session file this script
produces) to this repository or any git history -- both are equivalent to
a live login for the scraping account. Treat them like a password.
"""
import json
import os

from instaloader import Instaloader

COOKIE_JSON_PATH = os.environ.get("COOKIE_JSON_PATH", "./edge_cookie.json")
SESSION_OUTPUT_DIR = os.environ.get("SESSION_OUTPUT_DIR", ".")

with open(COOKIE_JSON_PATH, newline='') as jsonfile:
    cookies = json.load(jsonfile)

instaloader = Instaloader(max_connection_attempts=1)
for cookie_data in cookies:
    name = cookie_data.get('name')
    value = cookie_data.get('value')
    instaloader.context._session.cookies.update({name: value})

username = instaloader.test_login()
if not username:
    raise SystemExit("Not logged in. Is the cookie export still valid (are you logged in successfully in the browser)?")

instaloader.context.username = username
instaloader.save_session_to_file(os.path.join(SESSION_OUTPUT_DIR, f"session-{username}"))
print(f"Session saved for {username} in {SESSION_OUTPUT_DIR} -- set IG_SESSION_FILE to that path for the other scripts.")
