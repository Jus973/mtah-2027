#!/bin/sh
# Docker entry point (Dockerfile CMD): parse the ATD and build the submission for $TEAM (default:
# the team in configs/base.yaml), then copy the file to the mounted /out.
set -e
TEAM="${TEAM:-$(python -c 'from mtah import config; print(config.load()["team"])')}"
make PY=python cache submission TEAM="$TEAM"
mkdir -p /out
cp "submissions/final/$TEAM.csv" /out/
