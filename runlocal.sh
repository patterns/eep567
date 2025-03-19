#!/bin/sh


# also possible to activate manually
#source .venv/bin/activate
#python local.py

# and to purge
#uv cache clean

# command to run script
##uv run --with tensorflow local.py
##uv run --with tensorflow a2-15s.py
##uv run --with tensorflow a2-15s-novpn.py
##uv run --with tensorflow a2-60s.py
uv run --with tensorflow a2-30s.py

