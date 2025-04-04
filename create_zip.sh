#!/bin/sh

rm hands_on.zip
mkdir ai4biomed_hands_on
cp -rp data/ .python-version *.ipynb pyproject.toml README.md uv.lock install.html hands_on.html ai4biomed_hands_on
zip -r -0 hands_on.zip ai4biomed_hands_on/
rm -rf ai4biomed_hands_on
exit 0