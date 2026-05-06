#!/usr/bin/env bash
set -e
DEST="$(dirname "$0")/../models"
TMP=$(mktemp -d)

echo "Cloning MuJoCo Menagerie (sparse)..."
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$TMP/menagerie"
cd "$TMP/menagerie"
git sparse-checkout set kinova_gen3

echo "Copying kinova_gen3 to models/..."
mkdir -p "$DEST"
cp -r kinova_gen3 "$DEST/"

rm -rf "$TMP"
echo "Done. Model at models/kinova_gen3/"
