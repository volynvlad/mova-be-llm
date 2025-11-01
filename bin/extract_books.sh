#!/bin/sh

for dir in ./data/raw/*/; do
  echo "Entering directory: $dir"
  
  for zipfile in "$dir"/*; do
    if [[ -f "$zipfile" && "$zipfile" == *.zip ]]; then
      echo "Unzipping: $zipfile"
      file="${zipfile%.zip}"
      unzip "$zipfile" -d "$file"
    fi
  done
done

