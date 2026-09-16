"""
Re-scrapes the caption for each already-identified post (keycode) in a
filtered video dataset (e.g. dataset/ig_video_visualValid_N584_0901-1104_zscoreEngagement_0328.csv),
appending party/state/name/keycode/post_date/caption/caption_demojize rows
to an output CSV. Resumable: posts whose keycode is already in the output
CSV are skipped.

Uses instaloader with a saved browser session (see Environment below) --
NOT a plaintext password. See ig_get_video.py's docstring for the same
Terms-of-Service caveat; this script makes one Instagram request per row
in the source file, so it is even more rate-limit-sensitive.

Environment:
- IG_ACCOUNT_USERNAME: the Instagram username whose saved session to use.
- IG_SESSION_FILE: path to the session file produced by
  import_browser_session.py.
- SOURCE_CSV: the filtered video dataset providing keycode/party/state/name
  and a post_date column to restrict the date range.
- OUTPUT_CSV: where to append scraped captions.
"""
import os
import time
import emoji
import pandas as pd
from instaloader import Instaloader, Post, ConnectionException
from tqdm import tqdm

account = os.environ["IG_ACCOUNT_USERNAME"]
session_file = os.environ.get("IG_SESSION_FILE", f"./session-{account}")
start_date_str = os.environ.get("START_DATE", "2024/9/1")
end_date_str = os.environ.get("END_DATE", "2024/11/4")
save_file = os.environ.get("OUTPUT_CSV", "./output/ig_video_posts.csv")
source_file = os.environ.get("SOURCE_CSV", "../../dataset/ig_video_visualValid_N584_0901-1104_zscoreEngagement_0328.csv")

instagram = Instaloader(download_pictures=False, download_videos=False,
                        download_video_thumbnails=False, save_metadata=False, max_connection_attempts=0)

try:
    instagram.load_session_from_file(account, filename=session_file)
    username = instagram.test_login()
    if not username:
        raise ConnectionException()
except ConnectionException:
    raise SystemExit("Cookie import failed. Are you logged in successfully in Firefox?")

instagram.save_session_to_file()

if os.path.exists(save_file):
    df_existing = pd.read_csv(save_file, usecols=["keycode"])
    scraped_shortcodes = set(df_existing["keycode"].astype(str).unique())
else:
    scraped_shortcodes = set()

def scrape_data(shortcode, party, state, name):
    try:
        if shortcode in scraped_shortcodes:
            print(f"Skipping {shortcode} as already scraped.")
            return

        post = Post.from_shortcode(instagram.context, shortcode)
        time.sleep(3)

        caption_data = {
            "party": party,
            "state": state,
            "name": name,
            "keycode": post.shortcode,
            "post_date": post.date_utc.strftime('%Y-%m-%d'),
            "caption": post.caption if post.caption else "",
            "caption_demojize": emoji.demojize(post.caption) if post.caption else "",
        }

        df = pd.DataFrame([caption_data])

        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        if os.path.exists(save_file):
            df.to_csv(save_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(save_file, index=False, encoding='utf-8-sig')

        print(f"Done scraping caption of {shortcode}, saved to {save_file}.")
        scraped_shortcodes.add(shortcode)

    except Exception as e:
        print(f"Failed to scrape {shortcode}: {e}")

start_date = pd.to_datetime(start_date_str)
end_date = pd.to_datetime(end_date_str)

df = pd.read_csv(source_file)
df['post_date'] = pd.to_datetime(df['post_date'])
df_filtered = df[(df['post_date'] >= start_date) & (df['post_date'] <= end_date)]

df_filtered = df_filtered[~df_filtered["keycode"].astype(str).isin(scraped_shortcodes)]

for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
    scrape_data(row["keycode"], row["party"], row["state"], row["name"])
