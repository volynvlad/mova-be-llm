#!/bin/sh

find ./data/raw -type f -name "*.zip" | while read -r zipfile; do
  echo "Unzipping: $zipfile"
  
  outdir="${zipfile%.zip}"
  
  unzip -o "$zipfile" -d "$outdir"
done
