numbers_of_files=$(ls data/raw/knihi_books | wc -l)
number_of_zips=$(ls data/raw/knihi_books | grep zip | wc -l)

size_in_megabytes=$(du --block-size=1M data/raw -s | cut -f1)
size_in_gigabytes=$(du --block-size=1G data/raw -s | cut -f1)

size_in_megabytes_nozip=$(du --block-size=1M data/raw -s  --exclude="*.zip" | cut -f1)
size_in_gigabytes_nozip=$(du --block-size=1G data/raw -s  --exclude="*.zip" | cut -f1)

echo "Number of files: $numbers_of_files"
echo "Number of zip files: $number_of_zips"
echo "Size of data: ${size_in_megabytes}M or ${size_in_gigabytes}G"
echo "Size of data without .zip: ${size_in_megabytes_nozip}M or ${size_in_gigabytes_nozip}G"
