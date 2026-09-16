"""
Step 1 of the Instagram data-collection pipeline: given a spreadsheet of
candidate accounts (columns States/source/Party/name_id/ID -- ID is
resolved from name_id via get_ID_from_username on first run and cached),
downloads every video post published within [start_date, end_date] for
each account, saving the .mp4 files and a per-account
"{source}_posts.csv" with post_id/post_date/post_likes/post_comments/post_text.

Uses instaloader with a saved browser session (see Environment below) --
NOT a plaintext password. Scraping Instagram this way is against
Instagram's Terms of Service; this script is included for research
transparency (this is how the paper's raw video dataset was collected),
not as an endorsement -- rate limits and account bans are real risks, and
IG_ACCOUNT_USERNAME should be a disposable/secondary account, not a
personal one.

Environment:
- IG_ACCOUNT_USERNAME: the Instagram username whose saved session to use.
- IG_SESSION_FILE: path to the session file produced by
  import_browser_session.py (this script never touches a password).
- ACCOUNTS_XLSX: path to the account-list spreadsheet described above.
- OUTPUT_DIR: where to save downloaded videos and per-account CSVs.
- START_DATE / END_DATE ("YYYY-MM-DD"): date range to collect posts from;
  defaults to 2024-09-01 / 2024-11-30 (the range used for this paper).
"""
import instaloader
import time
import datetime
import pandas as pd
import argparse
import os
from random import randint

post_count = 1
have_post = 1
log_file = open("log.txt", "a", encoding='utf-8')  # Initialize log_file
log_file_error = open("error_log.txt", "a", encoding='utf-8')  # Initialize error log file

def get_ID_from_username(ig, L):
    username = ig.split("/")[-2]
    profile = instaloader.Profile.from_username(L.context, username)
    return str(profile.userid)

def scrape_post(source, name_id, start_date, end_date, post_iterator, range_s, range_e, to_process_pos, output_dir):
    global post_count, have_post
    posts_data = []

    L = instaloader.Instaloader()
    for post in post_iterator:
        n_sleep = randint(1, 3)
        time.sleep(n_sleep)
        if start_date.date() <= post.date.date() <= end_date.date():

            print(post.date.date())
            try :
                if (post.typename == ("GraphVideo" and "Reel") in post.caption) or post.is_video:
                    try:
                        post_id = post.shortcode
                        post_date = post.date
                        post_likes = post.likes
                        post_comments = post.comments
                        post_text = post.caption

                        link = f'https://www.instagram.com/p/{post_id}'
                        date_time = post_date.strftime('%Y-%m-%d')

                        # 儲存影片
                        video_filename = os.path.join(output_dir, f"{source}_{post.shortcode}.mp4")

                        # 使用 Instaloader 下載影片
                        L.download_post(post, target=video_filename)
                        print(f"影片已儲存至: {video_filename}")

                        data = (source, date_time, link, name_id, post_likes, post_comments, post_text)
                        print(data)

                        df = pd.DataFrame([{'source': source,
                                            'post_id': post_id,
                                            'post_date': post_date,
                                            'post_likes': post_likes,
                                            'post_comments': post_comments,
                                            'post_text': post_text}])
                        posts_data.append(df)
                    except Exception as e:
                        print(f"Error: {e}")

                    n_sleep = randint(3, 20)  # Random delay
                    print(f"...{post_count}...post...delay.....{n_sleep}s....{to_process_pos}@{range_s}~{range_e}\n")
                    time.sleep(n_sleep)
                    post_count += 1
            except Exception as e:
                print(f"Error: {e}")

    if not posts_data:
        have_post = 0
        print('No posts')
        log_file.write(f"No post\t{name_id}\n")
    else:
        have_post = 1
        log_file.write(f"{source}\t{name_id}\tcsv\n")

    if posts_data:
        all_posts_df = pd.concat(posts_data, ignore_index=True)
        all_posts_df.to_csv(os.path.join(output_dir, f"{source}_posts.csv"), index=False)

def main(account, session_file, accounts_xlsx, output_dir, startdate, enddate, range_s, range_e, have_ID=False):
    global log_file, log_file_error, post_count, have_post

    if startdate is None or enddate is None:
        startdate = datetime.datetime.today()  # Set start and end date to today
        enddate = datetime.datetime.today()

    os.makedirs(output_dir, exist_ok=True)

    print("Starting Instaloader...\n")
    L = instaloader.Instaloader()
    L.load_session_from_file(account, filename=session_file)

    df = pd.read_excel(accounts_xlsx)
    print(df.head())

    if not have_ID:
        df = df.rename(columns={"Instagram": 'name_id', "Name": 'source'}, errors="raise")
        df["ID"] = df['name_id'].apply(lambda x: get_ID_from_username(x, L=L))
        df.to_excel(accounts_xlsx)

    to_process_n = len(list(zip(df['source'], df['name_id'], df['ID'])))
    to_process_pos = 0
    user_count = 1

    # Check if range_s and range_e are None and set them to appropriate defaults
    if range_s is None:
        range_s = 0  # Default start range
    if range_e is None:
        range_e = float('inf')  # Default end range (no limit)

    for source, name_id, id in zip(df['source'], df['name_id'], df['ID']):
        aa_now = datetime.datetime.now()
        to_process_pos += 1

        # Apply the range filter
        if not (range_s == 0 and range_e == float('inf')):  # If range_s or range_e is set
            if to_process_pos < range_s or to_process_pos > range_e:
                continue

        print(f"Trying {name_id} at {aa_now} Position {to_process_pos} ({range_s}~{range_e})\n")

        try:
            if id != '404':
                profile = instaloader.Profile.from_id(L.context, id)
                post_iterator = profile.get_posts()
        except Exception as e:
            print(f"Error: {e}")
            continue

        scrape_post(source, name_id, startdate, enddate, post_iterator, range_s, range_e, to_process_pos, output_dir)

        if have_post == 1:
            n_sleep = randint(15, 30)  # Adjusted sleep interval
            print(f"{user_count} >> User, delay...{n_sleep}s\n")
            time.sleep(n_sleep)

        if user_count % 30 == 0:
            print(f"Reloading session...\n")
            try:
                L.load_session_from_file(account, filename=session_file)
            except Exception as e:
                n_sleep = randint(5, 15)
                print(f"Login error, delay.....{n_sleep}s\n")
                time.sleep(n_sleep)
                print("-------STOP-------")
                return
            n_sleep = randint(30, 70)
            print(f">>>>> {user_count} --- Resting for {n_sleep}s ----- {datetime.datetime.now()}")
            time.sleep(n_sleep)

        user_count += 1

    print("---END---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download Instagram video posts for a list of candidate accounts")
    args = parser.parse_args()

    account = os.environ["IG_ACCOUNT_USERNAME"]
    session_file = os.environ.get("IG_SESSION_FILE", f"./session-{account}")
    accounts_xlsx = os.environ.get("ACCOUNTS_XLSX", "./ig_get_videos_with_ID.xlsx")
    output_dir = os.environ.get("OUTPUT_DIR", "./data/ig_videos_raw")

    startdate = datetime.datetime.strptime(os.environ.get("START_DATE", "2024-09-01"), "%Y-%m-%d")
    enddate = datetime.datetime.strptime(os.environ.get("END_DATE", "2024-11-30"), "%Y-%m-%d")

    main(account, session_file, accounts_xlsx, output_dir, startdate, enddate, range_s=0, range_e=float('inf'), have_ID=True)
