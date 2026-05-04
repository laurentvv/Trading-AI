import os, datetime

cache_dir = r"D:\GIT\fork\Trading-AI\data_cache"
for f in sorted(os.listdir(cache_dir)):
    fp = os.path.join(cache_dir, f)
    if os.path.isfile(fp):
        stat = os.stat(fp)
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
        size_kb = stat.st_size / 1024
        print(f"{mtime.strftime('%Y-%m-%d %H:%M')} {size_kb:>8.1f} KB  {f}")
