"""Create (or recreate) the SQLite schema.

Run:  python -m scripts.init_db          (keeps existing DB, just ensures tables exist)
      python -m scripts.init_db --reset  (deletes the DB file first, for a clean rebuild)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config
from db import connect, init_schema


def main():
    if "--reset" in sys.argv and os.path.exists(Config.DATABASE):
        os.remove(Config.DATABASE)
        print("removed existing database ->", Config.DATABASE)
    os.makedirs(os.path.dirname(Config.DATABASE), exist_ok=True)
    conn = connect(Config.DATABASE)
    init_schema(conn)
    conn.close()
    print("schema ready ->", Config.DATABASE)


if __name__ == "__main__":
    main()
